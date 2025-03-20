import os
import glob
import pickle
import numpy as np

def extract_loss_values(directory_path, deg):
    """Extract loss values from all .pkl files in a directory and save to a numpy array."""
    
    loss_values = []
    pkl_files = []

    # Loop through all .pkl files in the directory
    for i in range(0, 27):

        pkl_path = os.path.join(directory_path, f'{deg}_deg_{i}_ex0.pkl')
        if os.path.exists(pkl_path):
            pkl_files.extend(glob.glob(os.path.join(directory_path, f'{deg}_deg_{i}_ex0.pkl')))
        
        else:
            print(f"warning: {pkl_path} does not exist")

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as file:
            data = pickle.load(file)
            
            # Check if 'loss' key exists in the data
            if 'loss' in data:
                loss_values.append(data['loss'].cpu())
            else:
                print(f"Warning: 'loss' key not found in file {pkl_file}")

    # Convert the list of loss values to a numpy array
    loss_array = np.array(loss_values)
    return loss_array


'''input_path = f'/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/rollouts/zero_shot/models_1'
loss_array = extract_loss_values(input_path, 35)

output_path = os.path.join(input_path, f"loss_35.npy")

# Save the numpy array to a file
np.save(output_path, loss_array)

# Print the numpy array
#print("Loss values:", loss_array)
print(f"saved {output_path}")'''

degrees = [20, 25, 35, 40]
models=[800]

for deg in degrees:
    for model in models:

        input_path = f'/scratch/10114/naveen_raj_manoharan/gns-reptile/rollouts/vanilla_fine_tuned/except_decoder/models_{deg}_1/model-{model}/'
        loss_array = extract_loss_values(input_path, deg)
        output_dir = f'/scratch/10114/naveen_raj_manoharan/gns-reptile/loss/vanilla_fine_tuned/except_decoder/models_1/deg_{deg}'

        output_path = os.path.join(output_dir, f"{model}.npy")

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        

        # Save the numpy array to a file
        np.save(output_path, loss_array)

        # Print the numpy array
        #print("Loss values:", loss_array)
        print(f"saved {output_path}, loss_{model}")

