#!/bin/bash

DEGREES=(20 25 35 40)
MODELS=(200 400 600 800 1000 2000 3000 4000 5000)

for DEG in "${DEGREES[@]}"; do 
    TRAIN_DATA_DIR=../gns-reptile/datasets/few_shot/1_traj/${DEG}_deg/
    TRAIN_MODEL_DIR=../gns-reptile/models/vanilla_fine_tuned/only_processor/models_${DEG}_1/
    TRAIN_LOG_DIR=logs/vanilla_fine_tuned/only_processor/models_${DEG}_1

    python3 -m gns.train_0 \
    --config-path ../ \
    --config-name config_post_reptile.yaml \
    data.path=$TRAIN_DATA_DIR \
    model.path=$TRAIN_MODEL_DIR \
    logging.tensorboard_dir=$TRAIN_LOG_DIR \

done

for MODEL in "${MODELS[@]}"; do

    for DEG in "${DEGREES[@]}"; do
        
        ROLLOUT_DATA_DIR=../gns-reptile-task-encoder/datasets/rollouts/${DEG}_deg/
        MODEL_DIR=../gns-reptile/models/vanilla_fine_tuned/only_processor/models_${DEG}_1/
        ROLLOUT_DIR="../gns-reptile/rollouts/vanilla_fine_tuned/only_processor/models_${DEG}_deg/model-${MODEL}/"

        python3 -m gns.train \
        --config-path ../ \
        --config-name config_rollout.yaml \
        data.path="$ROLLOUT_DATA_DIR" \
        model.path="$MODEL_DIR" \
        model.file="model-${MODEL}.pt" \
        model.train_state_file="train_state-${MODEL}.pt" \
        output.path="$ROLLOUT_DIR" \
        output.filename="${DEG}_deg"
        
    done

done