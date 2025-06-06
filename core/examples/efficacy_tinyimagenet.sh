#!/bin/bash

# Exit on error
set -e
set -x

# Configuration variables
DATASET="TinyImageNet"
ARCH="vit"
TASK="mixed"
LR=0.0001
TRAIN_EPOCHS=2
UNLEARN_METHOD="Scrub"
DEVICE_IDX=1   # system GPU index you want (change this to 1, 2, etc.)
CONFIG_PATH="../unlearn_config_vit.json"
RESULT_PATH="../attack/attack_inferences"
PROTECTED_PATH="../attack/attack_inferences/TinyImageNet/protected_samples.pt"
VULNERABLE_PATH="../attack/attack_inferences/TinyImageNet/vulnerable_samples.pt"
SAVED_PATH="../attack/attack_inferences/TinyImageNet/shadow_90_0_Scrub_unlearn_mixed.pth"



# Export CUDA_VISIBLE_DEVICES so the script only sees one GPU
export CUDA_VISIBLE_DEVICES=$DEVICE_IDX

# Inside Python, we always use cuda:0 (mapped to the masked GPU)
DEVICE="cuda:0"

# Define log filename with GPU index for clarity
LOG_FILE="shadows_${UNLEARN_METHOD}_${TASK}_${DATASET}_gpu${DEVICE_IDX}.log"

# Run the experiment
python ../unlearn_mia.py \
    --dataset "$DATASET" \
    --arch "$ARCH" \
    --task "$TASK" \
    --return_accuracy \
    --lr "$LR" \
    --train_epochs "$TRAIN_EPOCHS" \
    --unlearn_method "$UNLEARN_METHOD" \
    --device "$DEVICE" \
    --result_path "$RESULT_PATH" \
    --privacy_path "$PROTECTED_PATH" \
    --vulnerable_path "$VULNERABLE_PATH"\
    --saved_results "$SAVED_PATH" \
    --config_path "$CONFIG_PATH" 2>&1 | tee "$LOG_FILE"



# Check for errors
EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "Script failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
    exit $EXIT_CODE
fi

echo "Script completed successfully on GPU $DEVICE_IDX." | tee -a "$LOG_FILE"

