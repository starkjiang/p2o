"""
trainer.py — Training loop for all five preference-optimisation methods.
 
Training order within each epoch: datasets are visited sequentially in the
order they appear in the loaders dict (e.g. HH → SHP → UF → ...).
Each dataset's batches are domain-pure (no cross-dataset mixing).
The policy carries state across datasets within an epoch.
 
Public API
----------
train_model(method, ref_model, loaders, cfg, device)
    -> history dict
 
The returned history dict contains:
  batch_x, eval_batch_x — x-axis indices for plots
  loss, reward_accuracy, ...  — per-log-slot training metrics
  eval_{ds}_accuracy, eval_{ds}_margin, eval_{ds}_kl — per-dataset eval
  final_{ds} — end-of-training eval dicts for each dataset
  training_time — wall-clock seconds
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from p2o.config import Config
from p2o.losses import (
    compute_response_logprobs,
    dpo_loss, ipo_loss, kto_loss, p2o_loss, pkto_loss,
    evaluate,
)

VALID_METHODS = ("dpo", "ipo", "kto", "p2o", "pkto")


# History initialisation.

def make_history(total_batches: int, log_every: int, eval_every: int, dataset_names: list) -> Dict:
    n_log  = max(1, total_batches // log_every)
    n_eval = max(1, total_batches // eval_every)
    nan = lambda n: [float("nan")] * n
    hist = dict(
        batch_x = [log_every  * (i + 1) for i in range(n_log)],
        eval_batch_x = [eval_every * (i + 1) for i in range(n_eval)],
        # Training metrics (global batch index).
        loss = nan(n_log),
        reward_accuracy = nan(n_log),
        reward_margin = nan(n_log),
        clip_frac = nan(n_log),
        grad_norm = nan(n_log),
        chosen_reward = nan(n_log),
        rejected_reward = nan(n_log),
    )
    for ds in dataset_names:
        hist[f"eval_{ds}_accuracy"] = nan(n_eval)
        hist[f"eval_{ds}_margin"] = nan(n_eval)
        hist[f"eval_{ds}_kl"] = nan(n_eval)
    return hist


# Single optimisation step.

def _one_step(
    method: str,
    policy,
    ref_model,
    opt,
    batch: Dict,
    cfg: Config,
    device,
    batch_idx: int,
) -> Tuple[Dict | None, float]:
    """
    Execute one outer batch step for *method*.

    For P²O / PKTO this performs K inner proximal gradient steps while
    holding the snapshot mean log-probs fixed.

    Returns (metrics_dict, grad_norm), or (None, 0.0) on a non-finite loss.
    """
    metrics, gn = None, 0.0

    if method in ("p2o", "pkto"):
        # Take policy snapshot
        policy.eval()
        with torch.no_grad():
            _, old_mc = compute_response_logprobs(
                policy,
                batch["c_input_ids"].to(device),
                batch["c_attention_mask"].to(device),
                batch["c_response_start"],
                batch["c_n_resp"],
            )
            _, old_mr = compute_response_logprobs(
                policy,
                batch["r_input_ids"].to(device),
                batch["r_attention_mask"].to(device),
                batch["r_response_start"],
                batch["r_n_resp"],
            )
        old_mc, old_mr = old_mc.cpu(), old_mr.cpu()
        policy.train()

        # K proximal inner steps.
        for k in range(cfg.K_proximal):
            opt.zero_grad()
            if method == "p2o":
                loss, m = p2o_loss(
                    policy, ref_model, old_mc, old_mr,
                    batch, device,
                    cfg.beta, cfg.eps_clip, cfg.lam_kl,
                )
            else:
                loss, m = pkto_loss(
                    policy, ref_model, old_mc, old_mr,
                    batch, device,
                    cfg.beta, cfg.eps_clip, cfg.lam_kl,
                    cfg.kto_lambda_d, cfg.kto_lambda_u,
                )
            if not torch.isfinite(loss):
                print(f"{method.upper()} non-finite loss at b={batch_idx} k={k}")
                continue
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), cfg.max_grad_norm
            ).item()
            opt.step()
            metrics = m

    else:
        # DPO / IPO / KTO: single gradient step with no proximal inner loop.
        opt.zero_grad()
        if method == "dpo":
            loss, m = dpo_loss(policy, ref_model, batch, device, cfg.beta)
        elif method == "ipo":
            loss, m = ipo_loss(policy, ref_model, batch, device, cfg.beta, cfg.ipo_tau)
        else:  # kto
            loss, m = kto_loss(
                policy, ref_model, batch, device,
                cfg.beta, cfg.kto_lambda_d, cfg.kto_lambda_u,
            )
        if not torch.isfinite(loss):
            print(f"{method.upper()} non-finite loss at b={batch_idx}")
            return None, 0.0
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), cfg.max_grad_norm
        ).item()
        opt.step()
        metrics = m

    return metrics, gn


# ── Main training function ─────────────────────────────────────────────────────

def train_model(
    method: str,
    ref_model,
    loaders: Dict[str, Tuple[DataLoader, DataLoader]],
    cfg: Config,
    device,
) -> Dict:
    """
    Train *method* for cfg.n_epochs epochs, visiting datasets sequentially
    in the order they appear in *loaders*.
 
    Parameters
    ----------
    method : one of VALID_METHODS
    ref_model : frozen reference model
    loaders : dict mapping dataset short name -> (train_loader, eval_loader)
    cfg : Config
    device : torch device
 
    Returns the history dict populated with training and eval metrics.
    """
    assert method in VALID_METHODS, f"Unknown method '{method}'"

    dataset_names = list(loaders.keys())
    train_loaders = {k: v[0] for k, v in loaders.items()}
    eval_loaders  = {k: v[1] for k, v in loaders.items()}
 
    batches_ds = min(len(tl) for tl in train_loaders.values())
    n_ds = len(dataset_names)
    batches_ep  = batches_ds * n_ds
    tot_batches = batches_ep * cfg.n_epochs
 
    ds_display = " → ".join(d.upper() for d in dataset_names)
    print(f"\n{'═' * 60}")
    print(f"Training: {method.upper()}  (sequential: {ds_display})")
    print(f"{batches_ds} batches/ds × {n_ds} ds × {cfg.n_epochs} epochs = {tot_batches} total")
    print(f"{'═' * 60}")
 
    hist = make_history(tot_batches, cfg.log_every, cfg.eval_every, dataset_names)

    policy = (
        AutoModelForCausalLM.from_pretrained(cfg.model_name)
        .to(device)
        .train()
    )
    print(f"Params: {sum(p.numel() for p in policy.parameters()):,}")

    opt = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    warmup = max(1, int(tot_batches * cfg.warmup_ratio))
    sched  = get_cosine_schedule_with_warmup(opt, warmup, tot_batches)

    RUN_KEYS = [
        "loss", "reward_accuracy", "reward_margin",
        "clip_frac", "grad_norm", "chosen_reward", "rejected_reward",
    ]
    run = {k: 0.0 for k in RUN_KEYS}
    run_n = 0
    batch_idx = 0   # global counter across all datasets and epochs
    t0 = time.time()

    DS_LOADERS = [(ds, train_loaders[ds]) for ds in dataset_names]


    for epoch in range(cfg.n_epochs):
        for ds_name, ds_loader in DS_LOADERS:
            for batch in tqdm(
                ds_loader,
                desc=f"Ep {epoch + 1}/{cfg.n_epochs} [{method.upper()}|{ds_name.upper()}]",
                leave=False,
            ):
                batch_idx += 1

                metrics, gn = _one_step(
                    method, policy, ref_model, opt, batch, cfg, device, batch_idx
                )
                sched.step()

                if metrics is None:
                    continue
                metrics["grad_norm"] = gn
                for k in RUN_KEYS:
                    run[k] += metrics.get(k, 0.0)
                run_n += 1

                # ── Train log slot ────────────────────────────────────
                if batch_idx % cfg.log_every == 0:
                    slot = batch_idx // cfg.log_every - 1
                    if run_n > 0 and 0 <= slot < len(hist["loss"]):
                        avg = {k: v / run_n for k, v in run.items()}
                        for k in avg:
                            if k in hist:
                                hist[k][slot] = avg[k]
                        print(
                            f"  [{method.upper()}|{ds_name}] b={batch_idx:4d} | "
                            f"loss={avg['loss']:.4f} | "
                            f"acc={avg['reward_accuracy']:.3f} | "
                            f"margin={avg['reward_margin']:.4f} | "
                            f"clip={avg['clip_frac']:.3f} | "
                            f"gnorm={avg['grad_norm']:.2f} | "
                            f"t={time.time() - t0:.0f}s"
                        )
                    run = {k: 0.0 for k in RUN_KEYS}
                    run_n = 0

                # ── Eval slot — all held-out sets ───────────────────
                if batch_idx % cfg.eval_every == 0:
                    slot = batch_idx // cfg.eval_every - 1
                    eval_parts = []
                    for ev_ds, ev_loader in eval_loaders.items():
                        ev = evaluate(policy, ref_model, ev_loader, device, cfg.beta)
                        if 0 <= slot < len(hist[f"eval_{ev_ds}_accuracy"]):
                            hist[f"eval_{ev_ds}_accuracy"][slot] = ev["reward_accuracy"]
                            hist[f"eval_{ev_ds}_margin"][slot]   = ev["reward_margin"]
                            hist[f"eval_{ev_ds}_kl"][slot]       = ev["kl"]
                        eval_parts.append(
                            f"{ev_ds.upper()} acc={ev['reward_accuracy']:.3f} "
                            f"mg={ev['reward_margin']:.4f} KL={ev['kl']:.5f}"
                        )
                    print(
                        f"  [{method.upper()}] EVAL b={batch_idx:4d} [in {ds_name.upper()}] | "
                        + " | ".join(eval_parts)
                    )

    # Final eval after all epochs are done.
    print("  Final evaluation...")
    for ds, ev_loader in eval_loaders.items():
        final = evaluate(policy, ref_model, ev_loader, device, cfg.beta)
        hist[f"final_{ds}"] = final
        print(
            f"FINAL {ds.upper()}: acc={final['reward_accuracy']:.3f}  "
            f"mg={final['reward_margin']:.4f}  KL={final['kl']:.5f}"
        )
 
    hist["training_time"] = time.time() - t0
    hist["dataset_names"] = dataset_names
 
    print(
        f"Time: {hist['training_time']:.0f}s "
        f"({hist['training_time'] / 60:.1f} min)"
    )

    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return hist
