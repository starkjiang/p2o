"""
data.py — Dataset loading and tokenisation.

Three preference datasets, each with a separate train split and a
held-out eval split.  Training loaders are intentionally NOT pooled:
the training loop visits HH → SHP → UF sequentially within every epoch
so gradient signal is always domain-pure.

Public API
----------
build_loaders(cfg, tokenizer) -> (train_hh, train_shp, train_uf,
                                   eval_hh,  eval_shp,  eval_uf)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm.auto import tqdm

from p2o.config import Config


# Tokenization helper functions.

def _split_hh(text: str) -> Tuple[str, str]:
    """Split a full hh-rlhf conversation string into (prompt, response)."""
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        mid = len(text) // 2
        return text[:mid], text[mid:]
    return text[: idx + len(marker)].strip(), text[idx + len(marker) :].strip()


def _tok_pair(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
    max_prompt_length: int,
) -> Optional[Dict]:
    """
    Tokenise a (prompt, response) pair into a left-padded fixed-length tensor dict.

    Returns None if either side is too short to be useful (response < 4 tokens).
    """
    p_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=True,
    )["input_ids"]
    r_ids = tokenizer(
        " " + response,
        truncation=True,
        max_length=max_length - len(p_ids),
        add_special_tokens=False,
    )["input_ids"]

    if len(r_ids) < 4 or len(p_ids) < 1:
        return None

    full = (p_ids + r_ids)[:max_length]
    pad  = max_length - len(full)
    return {
        "input_ids":      [tokenizer.pad_token_id] * pad + full,
        "attention_mask": [0] * pad + [1] * len(full),
        "response_start": max(pad + len(p_ids), 1),
        "n_resp":         len(r_ids),
    }


def _tensorise(d: Dict) -> Dict:
    return {
        k: torch.tensor(v, dtype=torch.long) if isinstance(v, list) else v
        for k, v in d.items()
    }


def collate(batch: List[Dict]) -> Dict:
    """Collate a list of pair-dicts into batched tensors."""
    out: Dict = {}
    for key in batch[0]:
        if key == "source":
            continue
        vals = [item[key] for item in batch]
        out[key] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


# Dataset loaders.

def _load_hh(
    n: int,
    skip: int,
    tag: str,
    cfg: Config,
    tokenizer,
) -> List[Dict]:
    """
    Stream Anthropic/hh-rlhf train split, skip the first *skip* valid pairs,
    then collect *n* valid pairs.

    Each row has 'chosen' and 'rejected' as full conversation strings.
    A last-occurrence split on '\\n\\nAssistant:' separates prompt from response.
    """
    raw     = load_dataset(cfg.dataset_hh, split="train")
    data:   List[Dict] = []
    skipped = 0

    for row in tqdm(raw, desc=f"hh-rlhf[{tag}]", leave=False):
        if len(data) >= n:
            break
        chosen   = row.get("chosen",   "")
        rejected = row.get("rejected", "")
        if not chosen or not rejected:
            continue

        p_c, r_c = _split_hh(chosen)
        p_r, r_r = _split_hh(rejected)

        tw = _tok_pair(tokenizer, p_c, r_c, cfg.max_length, cfg.max_prompt_length)
        tl = _tok_pair(tokenizer, p_r, r_r, cfg.max_length, cfg.max_prompt_length)
        if tw is None or tl is None:
            continue

        if skipped < skip:
            skipped += 1
            continue

        data.append({
            **{f"c_{k}": v for k, v in _tensorise(tw).items()},
            **{f"r_{k}": v for k, v in _tensorise(tl).items()},
            "source": tag,
        })

    return data


def _load_shp(
    n: int,
    skip: int,
    tag: str,
    cfg: Config,
    tokenizer,
) -> List[Dict]:
    """
    Stream stanfordnlp/SHP train split.

    Fields used: history (prompt), human_ref_A / human_ref_B (candidates),
    labels (1 = A preferred, 0 = B preferred).
    """
    raw     = load_dataset(cfg.dataset_shp, split="train")
    data:   List[Dict] = []
    skipped = 0

    for row in tqdm(raw, desc=f"SHP[{tag}]", leave=False):
        if len(data) >= n:
            break
        prompt = row.get("history", "").strip()
        ref_a  = row.get("human_ref_A", "").strip()
        ref_b  = row.get("human_ref_B", "").strip()
        label  = row.get("labels", 1)           # 1 = A preferred
        if not prompt or not ref_a or not ref_b:
            continue

        chosen   = ref_a if label == 1 else ref_b
        rejected = ref_b if label == 1 else ref_a

        tw = _tok_pair(tokenizer, prompt, chosen,   cfg.max_length, cfg.max_prompt_length)
        tl = _tok_pair(tokenizer, prompt, rejected, cfg.max_length, cfg.max_prompt_length)
        if tw is None or tl is None:
            continue

        if skipped < skip:
            skipped += 1
            continue

        data.append({
            **{f"c_{k}": v for k, v in _tensorise(tw).items()},
            **{f"r_{k}": v for k, v in _tensorise(tl).items()},
            "source": tag,
        })

    return data


def _load_uf(
    n: int,
    skip: int,
    tag: str,
    cfg: Config,
    tokenizer,
) -> List[Dict]:
    """
    Stream openbmb/UltraFeedback train split.

    Schema: instruction (prompt), completions (list of dicts with keys
    'response' and 'overall_score' or 'score').
    Chosen  = completion with the highest score.
    Rejected = completion with the lowest score.
    Pairs where best == worst score are skipped.
    """
    raw     = load_dataset(cfg.dataset_uf, split="train")
    data:   List[Dict] = []
    skipped = 0

    def _score(c: Dict) -> float:
        for key in ("overall_score", "score"):
            v = c.get(key)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return 0.0

    for row in tqdm(raw, desc=f"UF[{tag}]", leave=False):
        if len(data) >= n:
            break
        prompt      = row.get("instruction", "").strip()
        completions = row.get("completions", [])
        if not prompt or len(completions) < 2:
            continue

        scored = [(_score(c), c.get("response", "").strip()) for c in completions]
        scored = [(s, r) for s, r in scored if r]
        if len(scored) < 2:
            continue

        best  = max(scored, key=lambda x: x[0])
        worst = min(scored, key=lambda x: x[0])
        if best[0] == worst[0]:
            continue

        tw = _tok_pair(tokenizer, prompt, best[1],  cfg.max_length, cfg.max_prompt_length)
        tl = _tok_pair(tokenizer, prompt, worst[1], cfg.max_length, cfg.max_prompt_length)
        if tw is None or tl is None:
            continue

        if skipped < skip:
            skipped += 1
            continue

        data.append({
            **{f"c_{k}": v for k, v in _tensorise(tw).items()},
            **{f"r_{k}": v for k, v in _tensorise(tl).items()},
            "source": tag,
        })

    return data


# Public API.

def build_loaders(cfg: Config, tokenizer) -> Tuple[
    DataLoader, DataLoader, DataLoader,   # train: hh, shp, uf
    DataLoader, DataLoader, DataLoader,   # eval:  hh, shp, uf
]:
    """
    Load all three datasets and return six DataLoaders.

    Train loaders are separate (NOT pooled) and individually shuffled.
    The training loop iterates them HH → SHP → UF within every epoch.
    Eval loaders are unshuffled and never seen during training.
    """
    N_TR = cfg.n_train_per_ds
    N_EV = cfg.n_eval_per_ds

    print(f"\nLoading HH-RLHF  ({N_TR} train + {N_EV} eval)...")
    hh_train = _load_hh(N_TR, skip=0,    tag="hh-train", cfg=cfg, tokenizer=tokenizer)
    hh_eval  = _load_hh(N_EV, skip=N_TR, tag="hh-eval",  cfg=cfg, tokenizer=tokenizer)

    print(f"Loading SHP      ({N_TR} train + {N_EV} eval)...")
    shp_train = _load_shp(N_TR, skip=0,    tag="shp-train", cfg=cfg, tokenizer=tokenizer)
    shp_eval  = _load_shp(N_EV, skip=N_TR, tag="shp-eval",  cfg=cfg, tokenizer=tokenizer)

    print(f"Loading UltraFeedback ({N_TR} train + {N_EV} eval)...")
    uf_train = _load_uf(N_TR, skip=0,    tag="uf-train", cfg=cfg, tokenizer=tokenizer)
    uf_eval  = _load_uf(N_EV, skip=N_TR, tag="uf-eval",  cfg=cfg, tokenizer=tokenizer)

    assert len(hh_eval)  > 0, "HH eval split is empty"
    assert len(shp_eval) > 0, "SHP eval split is empty"
    assert len(uf_eval)  > 0, "UF eval split is empty"

    print(f"\nDataset sizes:")
    print(f"  HH  : {len(hh_train):4d} train | {len(hh_eval):3d} eval")
    print(f"  SHP : {len(shp_train):4d} train | {len(shp_eval):3d} eval")
    print(f"  UF  : {len(uf_train):4d} train | {len(uf_eval):3d} eval")

    def _train_loader(dataset):
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate,
            drop_last=True,
        )

    def _eval_loader(dataset):
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate,
            drop_last=False,
        )

    return (
        _train_loader(hh_train),
        _train_loader(shp_train),
        _train_loader(uf_train),
        _eval_loader(hh_eval),
        _eval_loader(shp_eval),
        _eval_loader(uf_eval),
    )
