BASE_DIR='../gns-reptile-task-encoder/rollouts/few_shot/models_1/model-300000'
DEGREES=(20 25 35 40)
MODELS=(200 400 600 800 1000)

for MODEL in "${MODELS[@]}"; do
    for DEG in "${DEGREES[@]}"; do
        for IDX in {0..26}; do
            FILENAME_1=${DEG}_deg__${IDX}_ex0.gif
            FILENAME_2=${DEG}_deg_${IDX}_ex0.gif

            if [[ -f "$BASE_DIR/models_1_${DEG}_1/model-${MODEL}/${FILENAME_1}" ]]; then
                mv $BASE_DIR/models_1_${DEG}_1/model-${MODEL}/${FILENAME_1} $BASE_DIR/models_1_${DEG}_1/model-${MODEL}/$FILENAME_2
            
            else
                echo "Warning: File not found: $BASE_DIR/models_1_${DEG}_1/model-${MODEL}/$FILENAME_1"
            fi
        done
    done
done
