#!/bin/bash

# Define base paths
BASE_INPUT_PATH="/work2/10114/naveen_raj_manoharan/frontera/terramechanics/datasets/mpm_inputs"
#BASE_INPUT_PATH='/work2/10069/hassaniqbal209/frontera/reptile_dataset'
BASE_OUTPUT_PATH="../terramechanics/datasets/npz_files/"

# Iterate over angle ranges
for angle in 30; do
  for i in {0..29}; do
    #INPUT_PATH="${BASE_INPUT_PATH}/${angle}_deg_0/results/2d-sand-column/"
    INPUT_PATH="${BASE_INPUT_PATH}/${angle}_deg_${i}/results/2d-sand-column/"
    OUTPUT_PATH="${BASE_OUTPUT_PATH}/${angle}_deg_${i}.npz"

    # Check if input path exists before running the command
    if [ -d "$INPUT_PATH" ]; then
      echo "Processing $INPUT_PATH..."
      python3 ../gns-mpm/make_npz/convert_hd5_to_npz_2d.py --path="$INPUT_PATH" --dt=1 --output="$OUTPUT_PATH"
    else
      echo "Skipping $INPUT_PATH: Directory does not exist."
    fi
  done
done