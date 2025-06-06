#!/bin/bash

# Exit on error
set -e
set -x

mkdir -p ./logs

SHADOW_PATH="../core/attack/attack_inferences/WikiText103/shadow_6_attack_random_npo_gpt2.pth"
UNLEARN_METHOD="npo"
SFT_EPOCHS=5
UNLEARN_EPOCHS=15
LOG_FILE="./logs/mia_inference_${UNLEARN_METHOD}_sft${SFT_EPOCHS}_unlearn${UNLEARN_EPOCHS}.log"
TARGET_DATA_PATH="./data/WikiText-103-local/gpt2/selective_dataset_prefixed"

# Run the Python script
python mia_inference.py \
    --shadow_path "$SHADOW_PATH" \
    --unlearn_method "$UNLEARN_METHOD" \
    --sft_epochs "$SFT_EPOCHS" \
    --target_data_path "$TARGET_DATA_PATH" \
    --unlearn_epochs "$UNLEARN_EPOCHS" 2>&1 | tee "$LOG_FILE"


EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "Script failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi

echo "Script completed successfully." | tee -a "$LOG_FILE"