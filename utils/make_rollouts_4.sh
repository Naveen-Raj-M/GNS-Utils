DATA_DIR='../gns-reptile-task-encoder/datasets/rollouts_ex'
#MODEL_DIR_2='../gns-reptile-task-encoder/models/reptile_training/models_8_test_4/'
MODEL_DIR_1='/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/models/reptile_training/on_partially_trained/with_extra_dimension/model-500000/models_2/'
MODEL_DIR_2='/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/models/reptile_training/on_partially_trained/with_extra_dimension/model-500000/models_1/'
DEGREES_1=(20 40)
DEGREES_2=(20 40)
MODELS_1=(729000)
MODELS_2=(654000)


for MODEL in "${MODELS_1[@]}"; do

    for DEG in "${DEGREES_1[@]}"; do
        
        OUTPUT_DIR_1="../gns-reptile-task-encoder/rollouts/zero_shot/on_partially_trained/with_extra_dimension/model-500000/models_2/model-${MODEL}/"

        python3 -m gns_new.train \
        --config-path ../ \
        --config-name config_rollout.yaml \
        data.path="$DATA_DIR/${DEG}_deg/" \
        model.path="$MODEL_DIR_1" \
        model.file="model-${MODEL}.pt" \
        model.train_state_file="train_state-${MODEL}.pt" \
        output.path="$OUTPUT_DIR_1" \
        output.filename="${DEG}_deg"
        
    done

done


for MODEL in "${MODELS_2[@]}"; do

    for DEG_2 in "${DEGREES_2[@]}"; do

        OUTPUT_DIR_2="../gns-reptile-task-encoder/rollouts/zero_shot/on_partially_trained/with_extra_dimension/model-500000/models_1/model-${MODEL}/"
        python3 -m gns_new.train \
        --config-path ../ \
        --config-name config_rollout.yaml \
        data.path=$DATA_DIR/${DEG_2}_deg/ \
        model.path=$MODEL_DIR_2 \
        model.file=model-${MODEL}.pt \
        model.train_state_file=train_state-${MODEL}.pt \
        output.path=$OUTPUT_DIR_2 \
        output.filename=${DEG_2}_deg

    done

done