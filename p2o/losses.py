"""
losses.py — All preference-optimisation loss functions and the shared evaluator.

Every loss receives (policy, ref_model, batch, ...) and returns (loss, metrics_dict).
The metrics dict always contains at least:
  loss, reward_accuracy, reward_margin, chosen_reward, rejected_reward, clip_frac

Algorithm references
--------------------
DPO  : Rafailov et al. 2023  — https://arxiv.org/abs/2305.18290
IPO  : Azar et al. 2023      — https://arxiv.org/abs/2310.12036
KTO  : Ethayarajh et al. 2023 — https://arxiv.org/abs/2402.01306
P²O  : (proposed)
PKTO : (proposed, this work)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


# Core log-prob computation.

def compute_response_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_starts,
    n_resp_tokens,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single forward pass over a batch of (prompt + response) sequences.

    Returns
    -------
    sum_lp  : (B,)  sum of log π over response tokens
                    → implicit reward  h = β · (sum_lp_θ − sum_lp_ref)
    mean_lp : (B,)  mean per-token log π over response tokens
                    → proximal ratio   ρ = exp(mean_lp_θ − mean_lp_old)
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    lp_all = F.log_softmax(out.logits, dim=-1)          # (B, L, V)
    shift_lp = lp_all[:, :-1, :]                           # (B, L-1, V)
    targets = input_ids[:, 1:]                            # (B, L-1)
    tok_lp = shift_lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    B, Lm1 = tok_lp.shape
    device = input_ids.device
    sum_lp = torch.zeros(B, device=device)
    mean_lp = torch.zeros(B, device=device)

    for i in range(B):
        rs = response_starts[i]
        nr = n_resp_tokens[i]
        s, e = rs - 1, min(rs - 1 + nr, Lm1)
        mask = attention_mask[i, 1:].float()
        toks = tok_lp[i, s:e] * mask[s:e]
        n_v = mask[s:e].sum().clamp(min=1)
        sum_lp[i] = toks.sum()
        mean_lp[i] = toks.sum() / n_v

    return sum_lp, mean_lp


def _get_logprobs(policy, ref_model, batch, device):
    """Compute policy and reference log-probs for both chosen and rejected."""
    c_ids = batch["c_input_ids"].to(device)
    c_mask = batch["c_attention_mask"].to(device)
    r_ids = batch["r_input_ids"].to(device)
    r_mask = batch["r_attention_mask"].to(device)
    c_rs, c_nr = batch["c_response_start"], batch["c_n_resp"]
    r_rs, r_nr = batch["r_response_start"], batch["r_n_resp"]

    s_pi_c, m_pi_c = compute_response_logprobs(policy, c_ids, c_mask, c_rs, c_nr)
    s_pi_r, _ = compute_response_logprobs(policy, r_ids, r_mask, r_rs, r_nr)
    with torch.no_grad():
        s_rf_c, m_rf_c = compute_response_logprobs(ref_model, c_ids, c_mask, c_rs, c_nr)
        s_rf_r, _ = compute_response_logprobs(ref_model, r_ids, r_mask, r_rs, r_nr)

    return s_pi_c, s_pi_r, s_rf_c, s_rf_r, m_pi_c, m_rf_c


def _base_metrics(h_c: torch.Tensor, h_r: torch.Tensor) -> Dict:
    margin = h_c - h_r
    return dict(
        reward_accuracy=(margin > 0).float().mean().item(),
        chosen_reward=h_c.mean().item(),
        rejected_reward=h_r.mean().item(),
        reward_margin=margin.mean().item(),
        clip_frac=0.0,
    )


# DPO.

def dpo_loss(policy, ref_model, batch, device, beta: float):
    """
    Direct Preference Optimisation (Rafailov et al. 2023).

    L = -E[ log σ(β·(h⁺ − h⁻)) ]
    where  h = β·log(π_θ / π_ref)  is the implicit reward.
    """
    s_pi_c, s_pi_r, s_rf_c, s_rf_r, _, _ = _get_logprobs(policy, ref_model, batch, device)
    h_c = beta * (s_pi_c - s_rf_c)
    h_r = beta * (s_pi_r - s_rf_r)
    loss = -F.logsigmoid(h_c - h_r).mean()
    with torch.no_grad():
        m = dict(loss=loss.item(), **_base_metrics(h_c, h_r))
    return loss, m


# IPO.

def ipo_loss(policy, ref_model, batch, device, beta: float, tau: float):
    """
    Identity Preference Optimisation (Azar et al. 2023).

    L = E[ ((h⁺ − h⁻)/β − 1/(2τ))² ]

    Replaces the log-sigmoid with a squared loss targeting a fixed margin,
    reducing susceptibility to unbounded reward growth.
    """
    s_pi_c, s_pi_r, s_rf_c, s_rf_r, _, _ = _get_logprobs(policy, ref_model, batch, device)
    h_c = beta * (s_pi_c - s_rf_c)
    h_r = beta * (s_pi_r - s_rf_r)
    loss = (((h_c - h_r) / beta - 1.0 / (2.0 * tau)) ** 2).mean()
    with torch.no_grad():
        m = dict(loss=loss.item(), **_base_metrics(h_c, h_r))
    return loss, m


# KTO.

def kto_loss(policy, ref_model, batch, device, beta: float, lam_d: float, lam_u: float):
    """
    Kahneman–Tversky Optimisation (Ethayarajh et al. 2023).

    Applies a prospect-theory value function with a batch-estimated KL
    reference point z_ref.  The two branches are weighted independently:

    L = λ_D · E[1 − σ(h⁺ − z_ref)] + λ_U · E[1 − σ(z_ref − h⁻)]

    z_ref = E[ mean_lp_θ(y⁺) − mean_lp_ref(y⁺) ].clamp(0)
    """
    s_pi_c, s_pi_r, s_rf_c, s_rf_r, m_pi_c, m_rf_c = _get_logprobs(
        policy, ref_model, batch, device
    )
    h_c = beta * (s_pi_c - s_rf_c)
    h_r = beta * (s_pi_r - s_rf_r)
    z_ref = (m_pi_c - m_rf_c).mean().detach().clamp(min=0.0)
    loss = (
        lam_d * (1 - torch.sigmoid(h_c - z_ref)).mean()
        + lam_u * (1 - torch.sigmoid(z_ref - h_r)).mean()
    )
    with torch.no_grad():
        m = dict(loss=loss.item(), z_ref=z_ref.item(), **_base_metrics(h_c, h_r))
    return loss, m


# P²O.

def p2o_loss(
    policy,
    ref_model,
    old_mean_c: torch.Tensor,
    old_mean_r: torch.Tensor,
    batch,
    device,
    beta: float,
    eps: float,
    lam: float,
):
    """
    Proximal Preference Optimisation (proposed).

    Augments DPO with a PPO-style proximal clipping mechanism.
    The mean-token importance ratio ρ = exp(mean_lp_θ − mean_lp_old) is
    clipped to [1−ε, 1+ε]; the pessimistic min of raw and clipped surrogates
    is then used, mirroring PPO's trust-region objective.

    L = -E[ log σ( min(ρ·Δh, clip(ρ)·Δh) ) ] + λ · E[KL]_+

    where Δh = h⁺ − h⁻  and  KL = mean_lp_θ − mean_lp_ref.
    """
    s_pi_c, s_pi_r, s_rf_c, s_rf_r, m_pi_c, m_rf_c = _get_logprobs(
        policy, ref_model, batch, device
    )
    h_c = beta * (s_pi_c - s_rf_c)
    h_r = beta * (s_pi_r - s_rf_r)

    # Rejected mean log-prob needed for proximal ratio (not returned by _get_logprobs)
    _, m_pi_r = compute_response_logprobs(
        policy,
        batch["r_input_ids"].to(device),
        batch["r_attention_mask"].to(device),
        batch["r_response_start"],
        batch["r_n_resp"],
    )

    log_rho_c = (m_pi_c - old_mean_c.to(device).detach()).clamp(-5.0, 5.0)
    log_rho_r = (m_pi_r - old_mean_r.to(device).detach()).clamp(-5.0, 5.0)
    rho_c, rho_r = log_rho_c.exp(), log_rho_r.exp()
    rho_c_clip = rho_c.clamp(1 - eps, 1 + eps)
    rho_r_clip = rho_r.clamp(1 - eps, 1 + eps)

    d_raw  = rho_c * h_c - rho_r * h_r
    d_clip = rho_c_clip * h_c - rho_r_clip * h_r
    loss_p = -F.logsigmoid(torch.min(d_raw, d_clip)).mean()
    loss_kl = lam * (m_pi_c - m_rf_c).mean().clamp(min=0.0)
    loss = loss_p + loss_kl

    with torch.no_grad():
        cf = (
            ((rho_c < 1 - eps) | (rho_c > 1 + eps)).float().mean()
            + ((rho_r < 1 - eps) | (rho_r > 1 + eps)).float().mean()
        ) / 2
        base = _base_metrics(h_c, h_r)
        base["clip_frac"] = cf.item()
        m = dict(
            loss=loss.item(),
            loss_pref=loss_p.item(),
            loss_kl=loss_kl.item(),
            **base,
        )
    return loss, m


# PKTO.

def pkto_loss(
    policy,
    ref_model,
    old_mean_c: torch.Tensor,
    old_mean_r: torch.Tensor,
    batch,
    device,
    beta: float,
    eps: float,
    lam: float,
    lam_d: float,
    lam_u: float,
):
    """
    Proximal KTO (proposed).

    KTO's prospect-theory utility margins are substituted into P²O's
    clipped surrogate in place of the raw implicit rewards.

    Utility margins (centred on the adaptive KL baseline z_ref):
      u⁺ = h⁺ − z_ref   (chosen utility above KL baseline)
      u⁻ = z_ref − h⁻   (rejected utility below KL baseline)

    L = -E[ log σ( min(Δ_raw, Δ_clip) ) ] + λ · E[KL]_+

    where:
      Δ_raw  = λ_D · ρ⁺       · u⁺  −  λ_U · ρ⁻       · u⁻
      Δ_clip = λ_D · clip(ρ⁺) · u⁺  −  λ_U · clip(ρ⁻) · u⁻

    Limiting cases:
      ε → ∞, λ = 0  →  recovers vanilla KTO (no clipping, no extra KL term)
      λ_D = λ_U = 1, z_ref = 0  →  recovers P²O
    """
    s_pi_c, s_pi_r, s_rf_c, s_rf_r, m_pi_c, m_rf_c = _get_logprobs(
        policy, ref_model, batch, device
    )
    h_c = beta * (s_pi_c - s_rf_c)
    h_r = beta * (s_pi_r - s_rf_r)

    # Adaptive KL baseline (stop-gradient, clamped ≥ 0)
    z_ref = (m_pi_c - m_rf_c).mean().detach().clamp(min=0.0)

    # KTO utility margins
    u_c = h_c - z_ref   # desirable:   chosen above baseline
    u_r = z_ref - h_r   # undesirable: rejected below baseline

    # Proximal ratios (same mechanism as P²O)
    _, m_pi_r = compute_response_logprobs(
        policy,
        batch["r_input_ids"].to(device),
        batch["r_attention_mask"].to(device),
        batch["r_response_start"],
        batch["r_n_resp"],
    )
    log_rho_c = (m_pi_c - old_mean_c.to(device).detach()).clamp(-5.0, 5.0)
    log_rho_r = (m_pi_r - old_mean_r.to(device).detach()).clamp(-5.0, 5.0)
    rho_c, rho_r = log_rho_c.exp(), log_rho_r.exp()
    rho_c_clip = rho_c.clamp(1 - eps, 1 + eps)
    rho_r_clip = rho_r.clamp(1 - eps, 1 + eps)

    d_raw  = lam_d * rho_c * u_c - lam_u * rho_r * u_r
    d_clip = lam_d * rho_c_clip * u_c - lam_u * rho_r_clip * u_r
    loss_p  = -F.logsigmoid(torch.min(d_raw, d_clip)).mean()
    loss_kl =  lam * (m_pi_c - m_rf_c).mean().clamp(min=0.0)
    loss = loss_p + loss_kl

    with torch.no_grad():
        cf = (
            ((rho_c < 1 - eps) | (rho_c > 1 + eps)).float().mean()
            + ((rho_r < 1 - eps) | (rho_r > 1 + eps)).float().mean()
        ) / 2
        base = _base_metrics(h_c, h_r)
        base["clip_frac"] = cf.item()
        m = dict(
            loss=loss.item(),
            loss_pref=loss_p.item(),
            loss_kl=loss_kl.item(),
            z_ref=z_ref.item(),
            **base,
        )
    return loss, m


# Evaluator.

@torch.no_grad()
def evaluate(policy, ref_model, loader, device, beta: float) -> Dict:
    """
    Compute reward_accuracy, reward_margin, and token-mean KL over *loader*.

    Safe against empty loaders (returns zeros with a warning).
    """
    policy.eval()
    acc = mg_s = kl_s = n = 0

    for batch in loader:
        c_ids = batch["c_input_ids"].to(device)
        c_mask = batch["c_attention_mask"].to(device)
        r_ids = batch["r_input_ids"].to(device)
        r_mask = batch["r_attention_mask"].to(device)
        B = c_ids.shape[0]
        if B == 0:
            continue

        s_pi_c, m_pi_c = compute_response_logprobs(
            policy, c_ids, c_mask,
            batch["c_response_start"], batch["c_n_resp"],
        )
        s_pi_r, _ = compute_response_logprobs(
            policy, r_ids, r_mask,
            batch["r_response_start"], batch["r_n_resp"],
        )
        s_rf_c, m_rf_c = compute_response_logprobs(
            ref_model, c_ids, c_mask,
            batch["c_response_start"], batch["c_n_resp"],
        )
        s_rf_r, _ = compute_response_logprobs(
            ref_model, r_ids, r_mask,
            batch["r_response_start"], batch["r_n_resp"],
        )

        h_c = beta * (s_pi_c - s_rf_c)
        h_r = beta * (s_pi_r - s_rf_r)
        acc += (h_c - h_r > 0).float().sum().item()
        mg_s += (h_c - h_r).sum().item()
        kl_s += (m_pi_c - m_rf_c).sum().item()
        n += B

    policy.train()

    if n == 0:
        print("Evaluate(): loader was empty — returning zeros")
        return dict(reward_accuracy=0.0, reward_margin=0.0, kl=0.0)

    return dict(reward_accuracy=acc / n, reward_margin=mg_s / n, kl=kl_s / n)
