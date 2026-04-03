"""
data.py — Dataset loading and tokenisation.

Six preference datasets, each with a separate train split and a
held-out eval split.  Training loaders are intentionally NOT pooled:
the training loop visits datasets sequentially within every epoch
so gradient signal is always domain-pure.

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
        "input_ids": [tokenizer.pad_token_id] * pad + full,
        "attention_mask": [0] * pad + [1] * len(full),
        "response_start": max(pad + len(p_ids), 1),
        "n_resp": len(r_ids),
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
    raw = load_dataset(cfg.dataset_hh, split="train")
    data: List[Dict] = []
    skipped = 0

    for row in tqdm(raw, desc=f"hh-rlhf[{tag}]", leave=False):
        if len(data) >= n:
            break
        chosen = row.get("chosen",   "")
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
    raw = load_dataset(cfg.dataset_shp, split="train")
    data: List[Dict] = []
    skipped = 0

    for row in tqdm(raw, desc=f"SHP[{tag}]", leave=False):
        if len(data) >= n:
            break
        prompt = row.get("history", "").strip()
        ref_a = row.get("human_ref_A", "").strip()
        ref_b = row.get("human_ref_B", "").strip()
        label = row.get("labels", 1)           # 1 = A preferred
        if not prompt or not ref_a or not ref_b:
            continue

        chosen = ref_a if label == 1 else ref_b
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
    raw = load_dataset(cfg.dataset_uf, split="train")
    data: List[Dict] = []
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
        prompt = row.get("instruction", "").strip()
        completions = row.get("completions", [])
        if not prompt or len(completions) < 2:
            continue

        scored = [(_score(c), c.get("response", "").strip()) for c in completions]
        scored = [(s, r) for s, r in scored if r]
        if len(scored) < 2:
            continue

        best = max(scored, key=lambda x: x[0])
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


def _load_orca(
    n: int,
    skip: int,
    tag: str,
    cfg: Config,
    tokenizer,
) -> List[Dict]:
    """
    Stream Intel/orca_dpo_pairs train split.
    Fields: question (prompt), chosen (response), rejected (response).
    """
    raw = load_dataset(cfg.dataset_orca, split="train")
    data: List[Dict] = []
    skipped = 0
 
    for row in tqdm(raw, desc=f"Orca[{tag}]", leave=False):
        if len(data) >= n:
            break
        prompt   = row.get("question", "").strip()
        chosen   = row.get("chosen", "").strip()
        rejected = row.get("rejected", "").strip()
        if not prompt or not chosen or not rejected:
            continue
 
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
 
 
def _load_pku(
    n: int,
    skip: int,
    tag: str,
    cfg: Config,
    tokenizer,
) -> List[Dict]:
    """
    Stream PKU-Alignment/PKU-SafeRLHF train split.
    Fields: prompt, response_0, response_1, better_response_id (0 or 1).
    """
    raw = load_dataset(cfg.dataset_pku, split="train")
    data: List[Dict] = []
    skipped = 0
 
    for row in tqdm(raw, desc=f"PKU[{tag}]", leave=False):
        if len(data) >= n:
            break
        prompt = row.get("prompt", "").strip()
        r0 = row.get("response_0", "").strip()
        r1 = row.get("response_1", "").strip()
        better = row.get("better_response_id", None)
        if not prompt or not r0 or not r1 or better is None:
            continue
 
        chosen   = r0 if better == 0 else r1
        rejected = r1 if better == 0 else r0
 
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
 
 
def _load_ufb(
    n: int,
    skip: int,
    tag: str,
    cfg: Config,
    tokenizer,
) -> List[Dict]:
    """
    Stream argilla/ultrafeedback-binarized-preferences train split.
    Fields: instruction (prompt), chosen_response (str), rejected_response (str).
    """
    raw = load_dataset(cfg.dataset_ufb, split="train")
    data: List[Dict] = []
    skipped = 0
 
    for row in tqdm(raw, desc=f"UFB[{tag}]", leave=False):
        if len(data) >= n:
            break
        prompt   = row.get("instruction", "").strip()
        chosen   = row.get("chosen_response", "").strip()
        rejected = row.get("rejected_response", "").strip()
        if not prompt or not chosen or not rejected:
            continue
 
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

# Maps short name -> (loader_func, cfg_attr_for_hf_path, display_label)
DATASET_REGISTRY: Dict[str, Tuple] = {
  "hh":   (_load_hh,   "dataset_hh",   "HH-RLHF"),
  "shp":  (_load_shp,  "dataset_shp",  "SHP"),
  "uf":   (_load_uf,   "dataset_uf",   "UltraFeedback"),
  "orca": (_load_orca, "dataset_orca", "Orca DPO"),
  "pku":  (_load_pku,  "dataset_pku",  "PKU-SafeRLHF"),
  "ufb":  (_load_ufb,  "dataset_ufb",  "UF-Binarized"),
}
 
VALID_DATASETS = tuple(DATASET_REGISTRY.keys())


# Public API.


def build_loaders(
    cfg: Config,
    tokenizer,
    dataset_names: Optional[List[str]] = None,
) -> Dict[str, Tuple[DataLoader, DataLoader]]:
    """
    Load datasets and return DataLoaders.
 
    Parameters
    ----------
    cfg : Config
    tokenizer : PreTrainedTokenizer
    dataset_names : list of short names (e.g. ["hh", "shp", "uf", "orca"]).
                    Defaults to ["hh", "shp", "uf"] if None.
 
    Returns
    -------
    dict mapping dataset short name -> (train_loader, eval_loader)
    """
    if dataset_names is None:
        dataset_names = ["hh", "shp", "uf"]
 
    for name in dataset_names:
        assert name in DATASET_REGISTRY, (
            f"Unknown dataset '{name}'. Valid: {VALID_DATASETS}"
        )
 
    N_TR = cfg.n_train_per_ds
    N_EV = cfg.n_eval_per_ds
 
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
 
    loaders: Dict[str, Tuple[DataLoader, DataLoader]] = {}
 
    for ds_name in dataset_names:
        loader_fn, cfg_attr, display = DATASET_REGISTRY[ds_name]
        hf_path = getattr(cfg, cfg_attr)
 
        print(f"\nLoading {display} ({N_TR} train + {N_EV} eval) [{hf_path}]...")
        train_data = loader_fn(
            N_TR, skip=0, tag=f"{ds_name}-train", cfg=cfg, tokenizer=tokenizer,
        )
        eval_data = loader_fn(
            N_EV, skip=N_TR, tag=f"{ds_name}-eval", cfg=cfg, tokenizer=tokenizer,
        )
        assert len(train_data) > 0, f"{display} train split is empty"
        assert len(eval_data) > 0, f"{display} eval split is empty"
 
        print(f"  {display}: {len(train_data):4d} train | {len(eval_data):3d} eval")
 
        loaders[ds_name] = (_train_loader(train_data), _eval_loader(eval_data))
 
    print(f"\nDatasets loaded: {', '.join(dataset_names)}")
    return loaders
