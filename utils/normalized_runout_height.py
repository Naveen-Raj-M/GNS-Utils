import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def inspect_pkl_file(file_paths):
    """Extract normalized height and runout values from a .pkl file."""
    
    pred_norm_height_values = {}
    pred_norm_runout_values = {}
    gt_norm_height_values = {}
    gt_norm_runout_values = {}

    for file_path in sorted(file_paths):

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            predictions = data['predicted_rollout']
            ground_truth = data['ground_truth_rollout']

            ntime, _, _ = predictions.shape

            # Find the indices to remove in the first time step
            first_time_step = ground_truth[0]
            x_coords_first = first_time_step[:, 0]
            y_coords_first = first_time_step[:, 1]

            
            least_x_indices = np.argsort(x_coords_first)[:48]
            max_x_indices = np.argsort(x_coords_first)[-48:]
            min_y_indices = np.argsort(y_coords_first)[:48]

            remove_indices = np.concatenate((least_x_indices, max_x_indices, min_y_indices))
            remove_indices = np.unique(remove_indices)

            pred_norm_runout = []
            pred_norm_height = []
            gt_norm_runout = []
            gt_norm_height = []
            y_values = []

            for t in range(ntime):
                for dataset, runout_list, height_list in [
                    (predictions[t], pred_norm_runout, pred_norm_height),
                    (ground_truth[t], gt_norm_runout, gt_norm_height),
                ]:
                    
                    x_coords = np.delete(dataset[:, 0], remove_indices)
                    y_coords = np.delete(dataset[:, 1], remove_indices)

                    min_x = np.min(x_coords)
                    max_x = np.max(x_coords)
                    max_y = np.max(y_coords)
                    min_y = np.min(y_coords)

                    normalized_runout = ((max_x - min_x) - 0.3) / 0.3
                    normalized_height = (max_y - 0.3) / 0.3

                    y_values.append ((max_x, min_x))
                    runout_list.append(normalized_runout)
                    height_list.append(normalized_height)
            
        pred_norm_height_values[base_name] = pred_norm_height
        pred_norm_runout_values[base_name] = pred_norm_runout
        gt_norm_height_values[base_name] = gt_norm_height
        gt_norm_runout_values[base_name] = gt_norm_runout

    return pred_norm_height_values, pred_norm_runout_values, gt_norm_height_values, gt_norm_runout_values

def plot_normalized_values(timesteps, pred_values, gt_values, ylabel, title, filename):
    """Plot and save the normalized values."""
    plt.figure(figsize=(10, 6))

    color_dict = {
        0 : "blue",
        1 : "green",
        2 : "orange",
        3 : "black"
    }
    
    for i, key in enumerate(pred_values):
        plt.plot(timesteps, pred_values[key], color_dict[i], label=f'Predicted {key}')
        
    plt.plot(timesteps, gt_values[next(iter(gt_values))], 'red', label=f'Ground Truth')
    
    plt.xlabel('Timestep')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def process_all_pkl_files(directory_path):
    """Process all .pkl files in a directory."""

    # Loop through all directories in the directory_path
    for root, dirs, _ in os.walk(directory_path):
        for dir_name in dirs:

            print(f"Processing dir: {dir_name}")

            dir_path = os.path.join(root, dir_name)

            # Loop through all .pkl files in the directory
            pkl_files = glob.glob(os.path.join(dir_path, '*.pkl'))
            pred_norm_height, pred_norm_runout, gt_norm_height, gt_norm_runout = inspect_pkl_file(pkl_files)

            timesteps = np.arange(len(pred_norm_height[next(iter(pred_norm_height))]))
            filename = os.path.join(dir_path,f'{dir_name}_normalized_height.png')
            # Plot normalized height
            plot_normalized_values(
                timesteps,
                pred_norm_height,
                gt_norm_height,
                ylabel='Normalized Height',
                title=f'Normalized Height over Time ({dir_name})',
                filename=filename,
            )

            filename = os.path.join(dir_path,f'{dir_name}_normalized_runout.png')
            # Plot normalized runout
            plot_normalized_values(
                timesteps,
                pred_norm_runout,
                gt_norm_runout,
                ylabel='Normalized Runout',
                title=f'Normalized Runout over Time ({dir_name})',
                filename=filename,
            )

            
directory_path = '../gns-reptile/rollouts/indiv_traj'
process_all_pkl_files(directory_path)
