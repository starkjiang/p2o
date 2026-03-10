"""
trainer.py — Training loop for all five preference-optimisation methods.

Training order within each epoch: HH-RLHF → SHP → UltraFeedback.
Each dataset's batches are domain-pure (no cross-dataset mixing).
The policy carries state across datasets within an epoch.

Public API
----------
train_model(method, ref_model, train_loaders, eval_loaders, cfg, device)
    -> history dict

The returned history dict contains:
  batch_x, eval_batch_x — x-axis indices for plots
  loss, reward_accuracy, ...  — per-log-slot training metrics
  eval_hh_*, eval_shp_*, eval_uf_* — per-eval-slot held-out metrics
  final_hh, final_shp, final_uf — end-of-training eval dicts
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

def make_history(total_batches: int, log_every: int, eval_every: int) -> Dict:
    n_log  = max(1, total_batches // log_every)
    n_eval = max(1, total_batches // eval_every)
    nan = lambda n: [float("nan")] * n
    return dict(
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
        # Eval metrics — Eval-A HH-RLHF.
        eval_hh_accuracy = nan(n_eval),
        eval_hh_margin = nan(n_eval),
        eval_hh_kl = nan(n_eval),
        # Eval metrics — Eval-B SHP.
        eval_shp_accuracy = nan(n_eval),
        eval_shp_margin = nan(n_eval),
        eval_shp_kl = nan(n_eval),
        # Eval metrics — Eval-C UltraFeedback.
        eval_uf_accuracy = nan(n_eval),
        eval_uf_margin = nan(n_eval),
        eval_uf_kl = nan(n_eval),
    )


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
    train_loader_hh: DataLoader,
    train_loader_shp: DataLoader,
    train_loader_uf: DataLoader,
    eval_loader_hh: DataLoader,
    eval_loader_shp: DataLoader,
    eval_loader_uf: DataLoader,
    cfg: Config,
    device,
) -> Dict:
    """
    Train *method* for cfg.n_epochs epochs, visiting datasets in order:
        HH-RLHF → SHP → UltraFeedback  (within every epoch)

    Eval is run against all three held-out loaders every cfg.eval_every
    global batches and once more at the end of training.

    Returns the history dict populated with training and eval metrics.
    """
    assert method in VALID_METHODS, f"Unknown method '{method}'"

    batches_ds = min(
        len(train_loader_hh),
        len(train_loader_shp),
        len(train_loader_uf),
    )
    batches_ep  = batches_ds * 3
    tot_batches = batches_ep * cfg.n_epochs

    print(f"\n{'═' * 60}")
    print(f"Training: {method.upper()}  (sequential: HH → SHP → UF)")
    print(f"{batches_ds} batches/ds × 3 ds × {cfg.n_epochs} epochs = {tot_batches} total")
    print(f"{'═' * 60}")

    hist = make_history(tot_batches, cfg.log_every, cfg.eval_every)

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

    DS_LOADERS = [
        ("HH",  train_loader_hh),
        ("SHP", train_loader_shp),
        ("UF",  train_loader_uf),
    ]

    for epoch in range(cfg.n_epochs):
        for ds_name, ds_loader in DS_LOADERS:
            for batch in tqdm(
                ds_loader,
                desc=f"Ep {epoch + 1}/{cfg.n_epochs} [{method.upper()}|{ds_name}]",
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

                # ── Eval slot — all three held-out sets ───────────────
                if batch_idx % cfg.eval_every == 0:
                    slot = batch_idx // cfg.eval_every - 1
                    ev_hh = evaluate(policy, ref_model, eval_loader_hh,  device, cfg.beta)
                    ev_shp = evaluate(policy, ref_model, eval_loader_shp, device, cfg.beta)
                    ev_uf = evaluate(policy, ref_model, eval_loader_uf,  device, cfg.beta)
                    if 0 <= slot < len(hist["eval_hh_kl"]):
                        hist["eval_hh_accuracy"][slot]  = ev_hh["reward_accuracy"]
                        hist["eval_hh_margin"][slot]    = ev_hh["reward_margin"]
                        hist["eval_hh_kl"][slot]        = ev_hh["kl"]
                        hist["eval_shp_accuracy"][slot] = ev_shp["reward_accuracy"]
                        hist["eval_shp_margin"][slot]   = ev_shp["reward_margin"]
                        hist["eval_shp_kl"][slot]       = ev_shp["kl"]
                        hist["eval_uf_accuracy"][slot]  = ev_uf["reward_accuracy"]
                        hist["eval_uf_margin"][slot]    = ev_uf["reward_margin"]
                        hist["eval_uf_kl"][slot]        = ev_uf["kl"]
                    print(
                        f"  [{method.upper()}] EVAL b={batch_idx:4d} [in {ds_name}] | "
                        f"HH  acc={ev_hh['reward_accuracy']:.3f} "
                        f"mg={ev_hh['reward_margin']:.4f} KL={ev_hh['kl']:.5f} | "
                        f"SHP acc={ev_shp['reward_accuracy']:.3f} "
                        f"mg={ev_shp['reward_margin']:.4f} KL={ev_shp['kl']:.5f} | "
                        f"UF  acc={ev_uf['reward_accuracy']:.3f} "
                        f"mg={ev_uf['reward_margin']:.4f} KL={ev_uf['kl']:.5f}"
                    )

    # Final eval after all epochs are done.
    print("  Final evaluation...")
    final_hh = evaluate(policy, ref_model, eval_loader_hh,  device, cfg.beta)
    final_shp = evaluate(policy, ref_model, eval_loader_shp, device, cfg.beta)
    final_uf = evaluate(policy, ref_model, eval_loader_uf,  device, cfg.beta)

    print(
        f"FINAL HH: acc={final_hh['reward_accuracy']:.3f}  "
        f"mg={final_hh['reward_margin']:.4f}  KL={final_hh['kl']:.5f}"
    )
    print(
        f"FINAL SHP: acc={final_shp['reward_accuracy']:.3f}  "
        f"mg={final_shp['reward_margin']:.4f}  KL={final_shp['kl']:.5f}"
    )
    print(
        f"FINAL UF: acc={final_uf['reward_accuracy']:.3f}  "
        f"mg={final_uf['reward_margin']:.4f}  KL={final_uf['kl']:.5f}"
    )

    hist["final_hh"] = final_hh
    hist["final_shp"] = final_shp
    hist["final_uf"] = final_uf
    hist["training_time"] = time.time() - t0

    print(
        f"Time: {hist['training_time']:.0f}s "
        f"({hist['training_time'] / 60:.1f} min)"
    )

    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return hist
