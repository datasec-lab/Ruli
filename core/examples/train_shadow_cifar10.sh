#!/bin/bash

# Exit on error
set -e
set -x

DATASET="cifar10"
ARCH="resnet18"
TASK="mixed"
LR=0.1
TRAIN_EPOCHS=50
UNLEARN_METHOD="Scrub"
WEIGHT_DECAY=5e-4
DEVICE_IDX=1
CONFIG_PATH="../unlearn_config.json"
RESULT_PATH="../attack/attack_inferences"
PROTECTED_PATH="../attack/attack_inferences/cifar10/protected_samples.pt"
VULNERABLE_PATH="../attack/attack_inferences/cifar10/vulnerable_samples.pt"
#SAVED_PATH="../attack/attack_inferences/cifar10/shadows_90_0_Scrub_unlearn_mixed.pth"


export CUDA_VISIBLE_DEVICES=$DEVICE_IDX
DEVICE="cuda:0"

LOG_FILE="shadows_${UNLEARN_METHOD}_${TASK}_${DATASET}_gpu${DEVICE_IDX}.log"

# Run the experiment
python ../unlearn_mia.py \
    --dataset "$DATASET" \
    --train_shadow \
    --arch "$ARCH" \
    --task "$TASK" \
    --return_accuracy \
    --lr "$LR" \
    --train_epochs "$TRAIN_EPOCHS" \
    --unlearn_method "$UNLEARN_METHOD" \
    --shadow_num "$SHADOW_NUM" \
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

