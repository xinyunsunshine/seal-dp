#!/bin/bash
# chmod +x train_eval.sh
# ./train_eval.sh
# Ensure the script stops on error
set -e

# Define the command with parameters
CONFIG_DIR="config"
CONFIG_NAME="dp_square_mh.yaml"
SEED=42
DEVICE="cuda:0"
EPOCH_NUM=5


# HYDRA_RUN_DIR="data/outputs/$(date +%Y.%m.%d)/$(date +%H.%M.%S)_lr${LR}_sigma${X_SIGMA}_k${k}" #the old dir name
HYDRA_RUN_DIR="data/outputs/square/test0"
LOG_NAME="square_training_sh_test0"

# Overwrite flag (set to true to re-run evaluation)
OVERWRITE=false

# Run the training script
python train.py \
    --config-dir=$CONFIG_DIR \
    --config-name=$CONFIG_NAME \
    training.seed=$SEED \
    training.device=$DEVICE \
    logging.name="$LOG_NAME" \
    hydra.run.dir="$HYDRA_RUN_DIR"\
    training.num_epochs=$EPOCH_NUM \

# Save the directory path
echo "$HYDRA_RUN_DIR" > last_run_dir.txt


# Run evaluation script
# ./eval_nokp.sh "$HYDRA_RUN_DIR"
# Run evaluation script
if [ "$OVERWRITE" = true ]; then
    python eval_process_nokp.py "$HYDRA_RUN_DIR" --overwrite
else
    python eval_process_nokp.py "$HYDRA_RUN_DIR"
fi