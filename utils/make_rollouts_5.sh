DATA_DIR='../gns-reptile/datasets/rollouts'
MODEL_DIR_2='../gns-reptile-task-encoder/models/reptile_training/models_4/'
MODEL_DIR_3='../gns-reptile-task-encoder/models/reptile_training/models_5/'
DEGREES=(35 40)
MODELS_1=(600000)
MODELS_2=(375000 425000 450000 475000)

#for MODEL in "${MODELS_1[@]}"; do

#    for DEG in "${DEGREES[@]}"; do

#        OUTPUT_DIR_2="../gns-reptile-task-encoder/rollouts/zero_shot/models_4/model-${MODEL}/"

#        python3 -m gns.train \
#        --config-path ../ \
#        --config-name config_rollout.yaml \
#        data.path="$DATA_DIR/${DEG}_deg/" \
#        model.path="$MODEL_DIR_2" \
#       model.file="model-${MODEL}.pt" \
#        model.train_state_file="train_state-${MODEL}.pt" \
#        output.path="$OUTPUT_DIR_2" \
#        output.filename="${DEG}_deg"
        
#    done

#done

DEGREES_2=(20 25)

for MODEL in "${MODELS_2[@]}"; do

    for DEG_2 in "${DEGREES_2[@]}"; do

        OUTPUT_DIR_3="../gns-reptile-task-encoder/rollouts/zero_shot/models_5/model-${MODEL}/"

        python3 -m gns.train \
        --config-path ../ \
        --config-name config_rollout.yaml \
        data.path=$DATA_DIR/${DEG_2}_deg/ \
        model.path=$MODEL_DIR_3 \
        model.file=model-${MODEL}.pt \
        model.train_state_file=train_state-${MODEL}.pt \
        output.path=$OUTPUT_DIR_3 \
        output.filename=${DEG_2}_deg

    done

done