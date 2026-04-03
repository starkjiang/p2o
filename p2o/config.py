"""
config.py — All hyperparameters in one dataclass.

Override any field from the CLI via scripts/train.py --field value,
or instantiate Config(**overrides) directly in Python.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # Model.
    model_name: str = "gpt2"
    max_length: int = 128 # total tokens (prompt + response)
    max_prompt_length: int = 64 # prompt is capped at this many tokens

    # Datasets.
    dataset_hh: str = "Anthropic/hh-rlhf"
    dataset_shp: str = "stanfordnlp/SHP"
    dataset_uf: str = "openbmb/UltraFeedback"
    n_train_per_ds: int = 400 # training pairs drawn from each dataset
    n_eval_per_ds: int = 100 # eval pairs per dataset (held-out)

    dataset_orca: str = "Intel/orca_dpo_pairs"
    dataset_pku: str = "PKU-Alignment/PKU-SafeRLHF"
    dataset_ufb: str = "argilla/ultrafeedback-binarized-preferences"

    # Shared optimization hyperparameters.
    beta: float = 0.5 # implicit-reward temperature, which is KEY
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05 # fraction of total steps for LR warm-up
    batch_size: int = 8
    n_epochs: int = 2
    max_grad_norm: float = 1.0

    # IPO.
    ipo_tau: float = 0.1 # margin target = 1 / (2·τ)

    # KTO.
    kto_lambda_d: float = 1.0 # weight on desirable (chosen) branch
    kto_lambda_u: float = 1.0 # weight on undesirable (rejected) branch

    # P²O / PKTO.
    eps_clip: float = 0.15 # proximal clip radius ε
    lam_kl: float = 0.08 # explicit KL penalty coefficient λ
    K_proximal: int = 3 # inner proximal gradient steps per batch

    # Logging.
    log_every: int = 10 # log training metrics every N batches
    eval_every: int = 50 # run held-out eval every N batches

    # I/O.
    output_dir: str = "./outputs"
    seed: int = 42

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})
