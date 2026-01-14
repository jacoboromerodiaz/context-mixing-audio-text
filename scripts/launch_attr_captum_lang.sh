#!/bin/bash
#SBATCH --account=bsc88
#SBATCH --job-name=captum_ca
#SBATCH --output=/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/logs/%x_%j.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/logs/%x_%j.err
#SBATCH -q acc_debug
#SBATCH -c 80
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 00-00:45:00
#SBATCH --exclusive
#SBATCH --exclude=as04r2b01,as02r3b04,as07r1b02

# === entorno ===
source /gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/.venv/bin/activate

# === args comunes (equivalentes al SimpleNamespace) ===
CKPT="/gpfs/projects/bsc88/speech/mm_s2st/outputs/checkpoints/speech_salamandra/salamandrast-jacobo/stage2/s2_TA-7b-ins_mhubert_st-jacobo_v2/"
RESULTS_DIR="/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/results/s2_TA-7b-ins_mhubert_st-jacobo_v2/s2tt-cot/fleurs-iber"
ALIGNED_DIR="$RESULTS_DIR"
OUTPUT_DIR="/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/outputs/v6/"
ATTR_METHOD="shapley-values"

SPLITTER='(\\nTranslate to (?:.+?)<\|im_end\|>\\n<\|im_start\|>assistant\\n)'
LANGS="en_ca"

echo "Args:"
echo "  --checkpoint_path ${CKPT}"
echo "  --results_dir     ${RESULTS_DIR}"
echo "  --output_dir      ${OUTPUT_DIR}"
echo "  --attr_method     ${ATTR_METHOD}"
echo "  --splitter        ${SPLITTER}"
echo "  --chunk_size      1"
echo "  --no_special_tokens"
echo "  --attribution_kwargs {'n_samples': 25}"
echo "  --resume          (enabled)"

python /gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/src/main.py \
    --checkpoint-path "${CKPT}" \
    --results-dir "${RESULTS_DIR}" \
    --aligned-huberts-dir "${ALIGNED_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --attr-method "${ATTR_METHOD}" \
    --splitter "${SPLITTER}" \
    --chunk-size 1 \
    --no-special-tokens \
    --lang-pairs $LANGS \
    --attribution-kwargs '{"n_samples": 25}' \
    --resume

