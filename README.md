# P²O — Proximal Preference Optimisation

Benchmarking five offline preference-optimisation algorithms on GPT-2 (117M) across three human-feedback datasets.

| Method | Paper | Trust region | Loss shape |
|--------|-------|:---:|---|
| **DPO** | Rafailov et al. 2023 | ❌ | log-sigmoid |
| **IPO** | Azar et al. 2023 | ❌ | squared |
| **KTO** | Ethayarajh et al. 2023 | ❌ | prospect-theory |
| **P²O** | (proposed) | ✅ | clipped log-sigmoid |
| **PKTO** | (proposed) | ✅ | clipped prospect-theory |

**P²O** augments DPO with a PPO-style proximal clipping mechanism:

```
L = -E[ log σ( min(ρ·Δh, clip(ρ, 1±ε)·Δh) ) ] + λ·KL
```

where `ρ = exp(mean_t Δlog π)` is the mean-token importance ratio relative to
a per-batch policy snapshot, and `Δh = h⁺ − h⁻` is the implicit reward margin.

---

## Repo layout

```
p2o/
├── p2o/
│   ├── config.py     # Config dataclass — all hyperparameters
│   ├── data.py       # Dataset loaders (HH-RLHF, SHP, UltraFeedback)
│   ├── losses.py     # DPO / IPO / KTO / P²O / PKTO losses + evaluate()
│   ├── trainer.py    # Training loop (sequential per-dataset)
│   └── plot.py       # Publication-quality figures and result tables
├── scripts/
│   ├── train.py          # CLI entry point — run experiments
│   └── plot_results.py   # Re-plot from a saved results.json
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/your-username/p2o.git
cd p2o
pip install -e .
# or: pip install -r requirements.txt
```

### 2. Run all five methods (default config)

```bash
python scripts/train.py
```

Estimated runtime on a single T4 GPU: ~140 min total.

### 3. Run a single method

```bash
python scripts/train.py --methods p2o
```

### 4. Override hyperparameters

```bash
python scripts/train.py --methods dpo p2o pkto \
    --n_train_per_ds 200 \
    --n_epochs 3 \
    --eps_clip 0.10 \
    --output_dir ./my_outputs
```

### 5. Re-plot from saved results

```bash
python scripts/plot_results.py outputs/results.json
```

---

## Datasets

Each method trains on three datasets **sequentially** within every epoch
(HH-RLHF → SHP → UltraFeedback). Batches are never mixed across datasets.
Evaluation is always on separate held-out splits.

| Split | Dataset | Pairs | Role |
|-------|---------|------:|------|
| Train-A | `Anthropic/hh-rlhf` | 400 | 1st each epoch |
| Train-B | `stanfordnlp/SHP` | 400 | 2nd each epoch |
| Train-C | `openbmb/UltraFeedback` | 400 | 3rd each epoch |
| Eval-A | `Anthropic/hh-rlhf` | 100 | held-out |
| Eval-B | `stanfordnlp/SHP` | 100 | held-out |
| Eval-C | `openbmb/UltraFeedback` | 100 | held-out |

---

## Key metrics

| Metric | Description | Goal |
|--------|-------------|------|
| **Reward accuracy** | Fraction of eval pairs where policy prefers chosen over rejected | ↑ higher |
| **Reward margin** | Mean `h⁺ − h⁻` on eval pairs | ↑ higher |
| **Token-mean KL** | Mean per-token KL from reference model | ↓ lower |

---

## Extending to a new method

1. Add a `my_loss(policy, ref_model, batch, device, ...)` function in `p2o/losses.py`.
2. Add a dispatch branch in `_one_step()` in `p2o/trainer.py`.
3. Add `"my_method"` to `VALID_METHODS` in `p2o/trainer.py`.
4. Add a colour/marker/linestyle entry in `p2o/plot.py`.
5. Run: `python scripts/train.py --methods my_method`.
