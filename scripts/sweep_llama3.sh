#!/usr/bin/env bash
set -euo pipefail

# sweep
BETAS=(0.5)
LRS=(1e-4 5e-4 1e-3)
BATCH_SIZES=(4 8 10)
LAM_KLS=(0.08 0.005 0.1)
EPS_CLIPS=(0.5)

# Fixed settings
MODEL_NAME="meta-llama/Llama-3.2-1B"
MODEL_SHORT="Llama-3_model"
N_TRAIN=400
N_EVAL=100
N_EPOCHS=2
MAX_LENGTH=256
MAX_PROMPT_LENGTH=128

BASE_OUT="out_llama3"
mkdir -p "$BASE_OUT"

TOTAL=$(( ${#BETAS[@]} * ${#LRS[@]} * ${#BATCH_SIZES[@]} * ${#LAM_KLS[@]} * ${#EPS_CLIPS[@]} ))
RUN=0

for BETA in "${BETAS[@]}"; do
  for LR in "${LRS[@]}"; do
    for BS in "${BATCH_SIZES[@]}"; do
      for LAM_KL in "${LAM_KLS[@]}"; do
        for EPS_CLIP in "${EPS_CLIPS[@]}"; do
          RUN=$((RUN + 1))
          OUT_DIR="${BASE_OUT}/${MODEL_SHORT}/beta_${BETA}_lr_${LR}_bs_${BS}_lamkl_${LAM_KL}_epsclip_${EPS_CLIP}"

          echo "=================================================="
          echo "Run ${RUN}/${TOTAL}: beta=${BETA}  lr=${LR}  bs=${BS}  lam_kl=${LAM_KL}  eps_clip=${EPS_CLIP}"
          echo "Output directory: ${OUT_DIR}"
          echo "=================================================="

          python scripts/train.py \
              --model_name "${MODEL_NAME}" \
              --n_train_per_ds "${N_TRAIN}" \
              --n_eval_per_ds "${N_EVAL}" \
              --n_epochs "${N_EPOCHS}" \
              --beta "${BETA}" \
              --lr "${LR}" \
              --batch_size "${BS}" \
              --max_grad_norm 1.0 \
              --max_length "${MAX_LENGTH}" \
              --max_prompt_length "${MAX_PROMPT_LENGTH}" \
              --lam_kl "${LAM_KL}" \
              --eps_clip "${EPS_CLIP}" \
              --output_dir "${OUT_DIR}"

          echo "Finished run ${RUN}/${TOTAL}: beta=${BETA} lr=${LR} bs=${BS} lam_kl=${LAM_KL} eps_clip=${EPS_CLIP}"
          echo
        done
      done
    done
  done
done

echo "=================================================="
echo "All ${TOTAL} sweep runs completed."
echo "Results in: ${BASE_OUT}/${MODEL_SHORT}/"
echo "=================================================="
