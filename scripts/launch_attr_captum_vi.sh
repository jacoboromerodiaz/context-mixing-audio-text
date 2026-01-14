#!/bin/bash
#SBATCH --account=bsc88
#SBATCH --job-name=captum_v5
#SBATCH --output=/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/logs/%x_%A_%a.log
#SBATCH --error=/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/logs/%x_%A_%a.err
#SBATCH -q acc_bscls
#SBATCH -c 20
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -t 00-24:00:00
#SBATCH --exclusive
#SBATCH --exclude=as04r2b01,as02r3b04,as07r1b02
#SBATCH --array=0-5            

# === entorno ===
source /gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/.venv/bin/activate

# === lista de pares ===
LANG_PAIRS=(en_ca en_es en_pt en_fr en_it en_de)   
LANGS=${LANG_PAIRS[$SLURM_ARRAY_TASK_ID]}

# === rutas ===
CKPT="/gpfs/projects/bsc88/speech/mm_s2st/outputs/checkpoints/speech_salamandra/salamandrast-jacobo/stage2/s2_TA-7b-ins_mhubert_st-jacobo_v5/"
RESULTS_DIR="/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/results/s2_TA-7b-ins_mhubert_st-jacobo_v5/s2tt-cot/fleurs-test"
ALIGNED_DIR="/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/results/s2_TA-7b-ins_mhubert_st-jacobo_v5/s2tt-cot/fleurs-test-aligned_huberts"
OUTPUT_DIR="/gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/outputs/v5_2/"
ATTR_METHOD="shapley-values"

SPLITTER='(\\nTranslate to (?:.+?)<\|im_end\|>\\n<\|im_start\|>assistant\\n)'

mkdir -p "$RESULTS_DIR" "$OUTPUT_DIR"

echo "Task $SLURM_ARRAY_TASK_ID -> LANGS=$LANGS"

python /gpfs/projects/bsc88/speech/research/scripts/Jacobo/s2t/input_att/captum/src/main.py \
    --checkpoint-path "${CKPT}" \
    --results-dir "${RESULTS_DIR}" \
    --aligned-huberts-dir "${ALIGNED_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --attr-method "${ATTR_METHOD}" \
    --splitter "${SPLITTER}" \
    --chunk-size 1 \
    --no-special-tokens \
    --lang-pairs "$LANGS" \
    --attribution-kwargs '{"n_samples": 25}' \
    --resume
