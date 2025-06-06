#!/bin/bash

# Exit on error
set -e
set -x
mkdir -p ./logs


SFT_EPOCH=5
UNLEARN_METHOD="npo"
SHADOW_NUM=6
PREFIX_EPOCHS=1
UNLEARN_EPOCHS=15
DEVICE_IDX=1
TARGET_DATA_PATH="./data/WikiText-103-local/gpt2/selective_dataset_prefixed"


export CUDA_VISIBLE_DEVICES=$DEVICE_IDX
DEVICE="cuda:0"
LOG_FILE="attack_${UNLEARN_METHOD}_shadow${SHADOW_NUM}_sft${SFT_EPOCH}_prefix${PREFIX_EPOCHS}_gpu${DEVICE_IDX}.log"


python attack_main.py \
    --sft_epoch "$SFT_EPOCH" \
    --unlearn_epochs "$UNLEARN_EPOCHS" \
    --unlearn_method "$UNLEARN_METHOD" \
    --shadow_num "$SHADOW_NUM" \
    --prefix_epochs "$PREFIX_EPOCHS" \
    --target_data_path "$TARGET_DATA_PATH" \
    --device "$DEVICE" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "Script failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi

echo "Script completed successfully on GPU $DEVICE_IDX." | tee -a "$LOG_FILE"
