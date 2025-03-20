#!/bin/bash


#python3 -m gns.train_0 --config-path ../ --config-name config_post_reptile.yaml data.path=../gns-reptile/datasets/few_shot/1_traj/20_deg/ model.path=../gns-reptile/models/few_shot/models_4_20_2/ training.steps=5000 logging.tensorboard_dir=logs/few_shot/models_20_1_2/ training.learning_rate.initial=1e-6
#python3 -m gns.train_0 --config-path ../ --config-name config_post_reptile.yaml data.path=../gns-reptile/datasets/few_shot/1_traj/25_deg/ model.path=../gns-reptile/models/few_shot/models_4_25_2/ training.steps=5000 logging.tensorboard_dir=logs/few_shot/models_25_1_2/ training.learning_rate.initial=1e-6
python3 -m gns.train_0 --config-path ../ --config-name config_post_reptile.yaml data.path=../gns-reptile/datasets/few_shot/1_traj/35_deg/ model.path=../gns-reptile-task-encoder/models/few_shot/models_4/models_4_35_4/ training.steps=500 logging.tensorboard_dir=logs/few_shot/models_4_35_1_4/ training.learning_rate.initial=1e-6
python3 -m gns.train_0 --config-path ../ --config-name config_post_reptile.yaml data.path=../gns-reptile/datasets/few_shot/1_traj/40_deg/ model.path=../gns-reptile-task-encoder/models/few_shot/models_4/models_4_40_4/ training.steps=500 logging.tensorboard_dir=logs/few_shot/models_4_40_1_4/ training.learning_rate.initial=1e-6
python3 -m gns.train --config-path ../ --config-name config_post_reptile.yaml data.path=../gns-reptile/datasets/few_shot/1_traj/35_deg/ model.path=../gns-reptile-task-encoder/models/few_shot/models_4/models_4_35_4/ model.file=model-500.pt model.train_state_file=train_state-500.pt training.steps=4500 logging.tensorboard_dir=logs/few_shot/models_4_35_1_4/ training.learning_rate.initial=1e-7
python3 -m gns.train --config-path ../ --config-name config_post_reptile.yaml data.path=../gns-reptile/datasets/few_shot/1_traj/40_deg/ model.path=../gns-reptile-task-encoder/models/few_shot/models_4/models_4_40_4/ model.file=model-500.pt model.train_state_file=train_state-500.pt training.steps=4500 logging.tensorboard_dir=logs/few_shot/models_4_40_1_4/ training.learning_rate.initial=1e-7

#python3 -m gns.train_0 --config-path ../ --config-name config_post_reptile.yaml data.path=../gns-reptile-task-encoder/datasets/few_shot/20_deg/ model.path=../gns-reptile-task-encoder/models/few_shot/models_7_20_1/ 
#python3 -m gns.train_0 --config-path ../ --config-name config_post_reptile.yaml data.path=../gns-reptile-task-encoder/datasets/few_shot/25_deg/ model.path=../gns-reptile-task-encoder/models/few_shot/models_7_25_1/
#python3 -m gns.train_0 --config-path ../ --config-name config_post_reptile.yaml data.path=../gns-reptile-task-encoder/datasets/few_shot/35_deg/ model.path=../gns-reptile-task-encoder/models/few_shot/models_7_35_1/
#python3 -m gns.train_0 --config-path ../ --config-name config_post_reptile.yaml data.path=../gns-reptile-task-encoder/datasets/few_shot/40_deg/ model.path=../gns-reptile-task-encoder/models/few_shot/models_7_40_1/

# Base directory for the npz files
#BASE_DIR="../gns-reptile/datasets/npz_files/"

# Degrees to iterate over
DEGREES=(35 40)
MODELS=(200 400 600 800 1000 2000 3000 4000 5000)

for MODEL in "${MODELS[@]}"; do
    # Loop through each degree
    for DEG in "${DEGREES[@]}"; do
    # Loop through indices 0 to 26
    #for IDX in {0..26}; do
    #FILENAME="${DEG}_deg_${IDX}.npz"
    #TEST_FILENAME="test.npz"

    # Check if the file exists
    #if [[ -f "$BASE_DIR/$FILENAME" ]]; then
        # Move the current file to test.npz
        #mv "$BASE_DIR/$FILENAME" "$BASE_DIR/$TEST_FILENAME"

        # Run the Python training script
        #python3 -m gns.train --config-path ../ --config-name config_rollout.yaml output.filename="${DEG}_deg_${IDX}" model.file=model-"${MODEL}".pt model.train_state_file=train_state-"${MODEL}".pt output.path=../gns-reptile/rollouts/vanilla_fine_tuned/models_"${DEG}"_1/model-"${MODEL}"/ model.path=../gns-reptile/models/vanilla_fine_tuned/models_"${DEG}"_1/
        #python3 -m gns.train --config-path ../ --config-name config_rollout.yaml output.filename="${DEG}_deg_${IDX}"
        python3 -m gns.train --config-path ../ --config-name config_rollout.yaml data.path=../gns-reptile/datasets/rollouts/${DEG}_deg/ output.filename="${DEG}_deg" model.file=model-"${MODEL}".pt model.train_state_file=train_state-"${MODEL}".pt output.path=../gns-reptile-task-encoder/rollouts/few_shot/models_4/models_4_"${DEG}"_4/model-"${MODEL}"/ model.path=../gns-reptile-task-encoder/models/few_shot/models_4/models_4_"${DEG}"_4/
        #python3 -m gns.train --config-path ../ --config-name config_rollout.yaml data.path=../gns-reptile/datasets/rollouts/${DEG}_deg/ output.filename="${DEG}_deg" model.file=model-"${MODEL}".pt model.train_state_file=train_state-"${MODEL}".pt output.path=../gns-reptile-task-encoder/rollouts/few_shot/models_4/models_4_"${DEG}"_1/model-"${MODEL}"/ model.path=../gns-reptile-task-encoder/models/few_shot/models_6/models_6_"${DEG}"_1/
        #python3 -m gns.train --config-path ../ --config-name config_rollout.yaml data.path=../gns-reptile/datasets/npz_files/ output.filename="${DEG}_deg_${IDX}" model.file=model-"${MODEL}".pt model.train_state_file=train_state-"${MODEL}".pt output.path=../gns-reptile-task-encoder/rollouts/few_shot/models_7_"${DEG}"_1/model-"${MODEL}"/ model.path=../gns-reptile-task-encoder/models/few_shot/models_7_"${DEG}"_1/

        # Move test.npz back to the original file
        #mv "$BASE_DIR/$TEST_FILENAME" "$BASE_DIR/$FILENAME"
    #else
    #    echo "Warning: File not found - $BASE_DIR/$FILENAME"
    #fi
    #done
    done
done



#for MODEL in "${MODELS[@]}"; do
    # Loop through each degree
#    for DEG in "${DEGREES[@]}"; do
#        FROM_DIR="../gns-reptile-task-encoder/rollouts/few_shot/models_1_"${DEG}"_1/model-"${MODEL}"/loss_"${MODEL}".npy"
#        TO_DIR="../gns-reptile-task-encoder/loss/few_shot/"${DEG}"_deg/"${MODEL}".npy"
#        cp $FROM_DIR $TO_DIR
#    done
#done