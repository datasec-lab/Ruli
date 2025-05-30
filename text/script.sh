#!/bin/bash

# Exit on error
set -e
set -x

# Activate the environment if needed
# source ~/anaconda3/bin/activate myenv

# Define parameters
SFT_EPOCH=5
UNLEARN_METHOD="npo"
SHADOW_NUM=60
PREFIX_EPOCHS=1
UNLEARN_EPOCHS=15
DEVICE_IDX=1   # system GPU index you want (change this to 1, 2, etc.)

# Export CUDA_VISIBLE_DEVICES so the script only sees one GPU
export CUDA_VISIBLE_DEVICES=$DEVICE_IDX

# Inside Python, we always use cuda:0 (mapped to the masked GPU)
DEVICE="cuda:0"

# Define log filename with GPU index for clarity
LOG_FILE="attack_${UNLEARN_METHOD}_shadow${SHADOW_NUM}_sft${SFT_EPOCH}_prefix${PREFIX_EPOCHS}_gpu${DEVICE_IDX}.log"


# Run the attack script and save both stdout and stderr to the log file
python attack_main.py \
    --sft_epoch "$SFT_EPOCH" \
    --unlearn_epochs "$UNLEARN_EPOCHS" \
    --unlearn_method "$UNLEARN_METHOD" \
    --shadow_num "$SHADOW_NUM" \
    --prefix_epochs "$PREFIX_EPOCHS" \
    --device "$DEVICE" 2>&1 | tee "$LOG_FILE"

# Check exit code
# Check exit code
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "Script failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi

echo "Script completed successfully on GPU $DEVICE_IDX." | tee -a "$LOG_FILE"
