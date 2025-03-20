import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy import stats
from scipy.stats import sem, t

def load_loss_values(npy_files):
    """Load loss values from .npz files and keep track of filenames."""
    all_loss_values = []
    filenames = []

    for npy_file in npy_files:
        data = np.load(npy_file)
        loss_values = data
        
        all_loss_values.append(loss_values)
        filenames.append(os.path.splitext(os.path.basename(npy_file))[0])

    return np.array((all_loss_values), dtype=object), filenames

def plot_loss_comparison(all_loss_values, filenames, output_dir):
    """Plot a bar chart comparing errors at each element with filenames as labels."""
    num_files = all_loss_values.shape[0]
    indices = np.arange(1, 7)

    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    bar_width = 0.8 / num_files  # Adjust bar width based on the number of files
    for i in range(num_files):
        plt.bar(indices + i * bar_width, all_loss_values[i], width=bar_width, label=filenames[i])

    plt.xlabel('Element Index')
    plt.ylabel('Loss Value')
    plt.title('Comparison of Errors for Each Test (25 Deg Friction Angle)') # Replace '40' with the friction angle compared
    plt.xticks(indices + bar_width * (num_files - 1) / 2, indices)
    plt.legend()
    plt.grid(True)

    output_file = os.path.join(output_dir,'loss_comparison_each_test_25_1.png')
    # Save the plot as a PNG file
    plt.savefig(output_file) # Replace '40' with the friction angle compared

    # Show the plot
    plt.show()

def calculate_confidence_interval(data, confidence=0.95):
    """Calculate the mean and 95% confidence interval for the data."""
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)  # Margin of error
    return mean, h

def plot_error_bar(loss_list, filenames_list, output_directory):
    """Plot a 2x2 grid comparing errors with 95% confidence intervals for different angles."""
    angles = [40]
    num_angles = len(angles)

    fig, axes = plt.subplots(1, 1, figsize=(12, 12))  # 1x1 grid
    #axes = axes.flatten()  # Flatten axes for easy indexing

    for i in range(min(len(loss_list), num_angles)):
        means = []
        conf_intervals = []

        # Calculate means and confidence intervals for each file
        #for loss_values in loss_list[i]:
        for j in range(len(filenames_list)):
            mean, conf_interval = calculate_confidence_interval(loss_list[j])
            means.append(mean)
            conf_intervals.append(conf_interval)

        indices = np.arange(len(means))

        # Plot bar chart with error bars
        axes.bar(indices, means, yerr=conf_intervals, capsize=5, alpha=0.7, label='Mean Loss with 95% CI')
        axes.set_yscale('log')
        axes.set_ylim(1e-5, 1e-2)
        #axes.set_xlabel('Training Steps')
        axes.set_ylabel('Loss Value (log scale)')
        axes.set_title(f'Comparison of Mean Loss for {angles[i]}° Friction Angle')
        axes.set_xticks(indices)
        print(filenames_list)
        axes.set_xticklabels(filenames_list, rotation=45, ha='right')
        axes.legend()
        axes.grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    output_file = os.path.join(output_directory, 'error_bar.png')
    plt.savefig(output_file, dpi=300)
    plt.show()

def plot_error_bar_comparison(directories: list[list[str]]):
    """
    Create a 2x2 grid of subplots, where each subplot compares the error bars of directories.

    Parameters:
        directories (list[list[str]]): A list of lists, where each sublist contains directories to compare in one subplot.
    """
    # Define training steps
    training_steps = [
        [0, 1000, 2000, 3000, 4000, 5000],
        [0, 1000, 2000, 3000, 4000, 5000]
    ]

    # Initialize the plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()  # Flatten the 2x2 grid for easier indexing
    
    #fig, axes = plt.subplots(1, 1, figsize=(12, 10))


    # Iterate through each group of directories and plot comparisons
    for subplot_idx, group in enumerate(directories):
        results = {}

        # Collect data for each directory in the group
        for i, directory in enumerate(group):
            means = []
            cis = []
            for step in training_steps[i]:
                file_path = os.path.join(directory, f"{step}.npy")
                data = np.load(file_path)  # Load data
                mean = np.mean(data)
                confidence_interval = t.ppf(0.975, len(data) - 1) * sem(data)  # 95% CI
                means.append(mean)
                cis.append(confidence_interval)
            
            if i == 1:
                name = 'Encoder'
            if i == 0:
                name = 'vanilla fine-tuned'
            results[name] = (means, cis)

        # Plot the results in the current subplot
        for label, (means, cis) in results.items():
            axs[subplot_idx].errorbar(
                training_steps[i], means, yerr=cis, label=label, capsize=5, marker='o', linestyle='-'
            )

            #axes.errorbar(
            #    training_steps[i], means, yerr=cis, label=label, capsize=5, marker='o', linestyle='-'
            #@)
        
        titles = ['20_deg', '25_deg', '35_deg', '40_deg' ]

        # Customize the subplot
        axs[subplot_idx].set_title(f"Comparison {titles[subplot_idx]}")
        axs[subplot_idx].set_yscale('log')
        axs[subplot_idx].set_ylim(1e-5, 1e-2)
        axs[subplot_idx].set_xlabel("Training Steps")
        axs[subplot_idx].set_ylabel("MSE Loss")
        axs[subplot_idx].legend()
        axs[subplot_idx].grid(True)

        '''axes.set_title(f"Comparison of Reptile training")
        axes.set_yscale('log')
        axes.set_ylim(1e-5, 1e-2)
        axes.set_xlabel("Training Steps")
        axes.set_ylabel("MSE Loss")
        axes.legend()
        axes.grid(True)'''

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(directories[0][0], 'loss_comparison_0-5000.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_loss_with_ci(directories: list[list[str]]):
    """
    Create an error bar chart with shaded confidence intervals.

    Parameters:
        directories (list[list[str]]): A list of lists, where each sublist contains directories to compare.
    """
    # Define training steps
    training_steps = [
        [0, 200, 400, 600, 800, 1000],
        [0, 200, 400, 600, 800, 1000]
    ]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Iterate through each group of directories and plot comparisons
    for i, group in enumerate(directories):
        results = {}

        # Collect data for each directory in the group
        for j, directory in enumerate(group):

            means = []
            lower_bound = []
            upper_bound = []
            
            for step in training_steps[j]:
                file_path = os.path.join(directory, f"{step}.npy")
                data = np.load(file_path)  # Load data
                mean = np.mean(data)
                ci = t.ppf(0.95, len(data) - 1) * sem(data)  # 90% Confidence Interval
                means.append(mean)
                lower_bound.append(mean - ci)
                upper_bound.append(mean + ci)
            
            label = '20° Friction Angle' if j == 0 else '40° Friction Angle'
            color = 'tab:blue' if j == 0 else 'tab:orange'
            
            # Plot solid line for mean
            ax.plot(training_steps[j], means, label=label, linestyle='-', color=color)
            # Plot shaded confidence interval
            ax.fill_between(training_steps[j], lower_bound, upper_bound, color=color, alpha=0.1)

    # Customize the plot
    #ax.set_title("Reptile Trained Model's Adaptation to Different Friction Angles", fontsize=20)
    ax.set_yscale('log')
    ax.set_ylim(7e-5, 3e-3)
    ax.set_xlabel("Training Steps", fontsize=40)
    ax.set_ylabel("MSE Loss", fontsize=40)
    ax.set_xticks([0, 1000, 2000])
    ax.set_yticks([1e-4, 1e-3])
    ax.tick_params(axis='both', labelsize=30)
    ax.legend(fontsize=40)
    ax.grid(True)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(directories[0][0], 'loss_comparison_ci.png'), dpi=300, bbox_inches='tight')


loss_list = []
filenames_list = []

'''angles = [20, 25]

for angle in angles:

    directory_path = f'/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/loss/zero_shot/models_5/{angle}_deg/'
   
    # Get all .npy files in the directory
    npy_files = sorted(glob.glob(os.path.join(directory_path, '*.npy')))

    # Custom order for sorting
    #custom_order = ["vanilla", "zero_shot", "vanilla_fine_tuned", "few_shot"]

    # Get all .npy files in the directory
    #npy_files = sorted(
    #    npy_files, 
    #    key=lambda x: custom_order.index(os.path.basename(x).split('.')[0]) if os.path.basename(x).split('.')[0] in custom_order else float('inf')
    #)

    # Load loss values from .npz files
    loss, filenames = load_loss_values(npy_files)

    loss_list.append(loss)
    filenames_list.append(filenames)

output_dir = '/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/loss/zero_shot/models_5/'


plot_error_bar(loss_list, filenames_list, output_dir)

directory_path = '/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/loss/zero_shot/models_8_test_4'


# Custom order for sorting
custom_order = ["vanilla", "Encoder_80", "All_80", "Encoder_325000", "All_325000"]

# Get all .npy files in the directory
npy_files = sorted(glob.glob(os.path.join(directory_path, '*.npy')))

# Sort files based on the custom order
npy_files_sorted = sorted(
    npy_files, 
    key=lambda x: custom_order.index(os.path.basename(x).split('.')[0]) if os.path.basename(x).split('.')[0] in custom_order else float('inf')
)
# Load loss values from .npz files
loss, filenames = load_loss_values(npy_files_sorted)

# Plot the comparison of errors
# plot_loss_comparison(all_loss_values, filenames, directory_path)

plot_error_bar(loss, filenames, directory_path)'''


#directory_1 = '/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/loss/few_shot/models_1_2/20_deg'
directory_2 = '../gns-reptile/loss/vanilla_fine_tuned/except_decoder/models_1/20_deg'
#directory_3 = '/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/loss/few_shot/models_1_2/25_deg'
directory_4 = '../gns-reptile/loss/vanilla_fine_tuned/except_decoder/models_1/25_deg'
#directory_5 = '/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/loss/few_shot/models_1_2/35_deg'
directory_6 = '../gns-reptile/loss/vanilla_fine_tuned/except_decoder/models_1/35_deg'
#directory_7 = '/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/loss/few_shot/models_1_2/40_deg'
directory_8 = '../gns-reptile/loss/vanilla_fine_tuned/except_decoder/models_1/40_deg'

directories = [[directory_2], 
                [directory_4],
               [directory_6],
               [ directory_8]]

#directories = [[directory_1, directory_3]]


plot_error_bar_comparison(directories)

#directory_1 = "/scratch1/10114/naveen_raj_manoharan/gns-reptile-task-encoder/loss/few_shot/models_1_2/20_deg"
#directory_2 = "/scratch1/10114/naveen_raj_manoharan/gns-reptile-task-encoder/loss/few_shot/models_1_2/40_deg"
#directories = [[directory_1, directory_2]]
#plot_loss_with_ci(directories)