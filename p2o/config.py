"""
config.py — All hyperparameters in one dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # Model.
    model_name: str = "gpt2"
    max_length: int = 128
    max_prompt_length: int = 64

    # Datasets.
    dataset_hh: str = "Anthropic/hh-rlhf"
    dataset_uf: str = "openbmb/UltraFeedback"
    n_train_per_ds: int = 400
    n_eval_per_ds: int = 100

    dataset_orca: str = "Intel/orca_dpo_pairs"

    # Shared optimization hyperparameters.
    beta: float = 0.5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    batch_size: int = 8
    n_epochs: int = 2
    max_grad_norm: float = 1.0

    # IPO.
    ipo_tau: float = 0.1

    # KTO.
    kto_lambda_d: float = 1.0
    kto_lambda_u: float = 1.0

    # P²O / PKTO.
    eps_clip: float = 0.15
    lam_kl: float = 0.08
    K_proximal: int = 3

    # Logging.
    log_every: int = 10
    eval_every: int = 50

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
