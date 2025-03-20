import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import ConnectionPatch

# Function to categorize the parameter based on the key
def categorize_key(key, type):

    if type == "entire model":
        # Dictionary to map substrings to categories
        category_map = {
            "_particle_type_embedding": "Embedding",
            "_encoder": "Encoder",
            "_processor": "Processor",
            "_decoder": "Decoder",
        }
    
    else:
        # Dictionary to map substrings to categories
        category_map = {
            "node_": "Node",
            "edge_": "Edge"
        }

    # Iterate through the category map to find the first matching substring
    for substring, category in category_map.items():
        if substring in key:
            return category
    
    # Default category if no match is found
    return "Other"


def make_bar_chart(top_100, outputs, type):
    # Extract data for the bar chart
    xtick_labels = [f"{categorize_key(item[0], type)}({item[1]})" for item in top_100]
    values_model1 = [item[3] for item in top_100]         # Values from model1
    values_model2 = [item[4] for item in top_100]         # Values from model2

    # Define bar width and positions
    x = np.arange(len(xtick_labels))  # Positions for xticks
    bar_width = 0.4

    # Create the bar chart
    fig = plt.figure(figsize=(20, 10))
    plt.bar(x - bar_width / 2, values_model1, bar_width, label="Vanilla GNS", color='blue')
    plt.bar(x + bar_width / 2, values_model2, bar_width, label="Reptile trained GNS", color='orange')

    # Customize the plot
    plt.xlabel("Parameter (Category(Index))", fontsize=12)
    plt.ylabel("Parameter Value", fontsize=12)
    plt.title(outputs[1], fontsize=14)
    plt.xticks(x, xtick_labels, rotation=90, fontsize=12)  # Rotate x-tick labels for better visibility
    plt.ylim(-2, 0.6)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(outputs[0], dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_pie_chart(model1, model2, outputs, types):

    top_1000 = True

    # Initialize a dictionary to hold total percentage differences for each category
    category_differences = defaultdict(float)
    # Temporary list to hold key-wise differences
    differences = []

    # Iterate through all keys in the models
    for key in model1.keys():
        if key in model2:  # Ensure the key exists in both models
            # Get parameters from both models
            param1 = model1[key].flatten().cpu().numpy()
            param2 = model2[key].flatten().cpu().numpy()
            
            if types[1] == "relative":
                diff_tensor = abs(param1 - param2).sum() / (abs(param1).sum() + 1e-8) * 100  # Avoid division by zero
            
            else:
                diff_tensor = abs(param1 - param2).sum() #absolute difference
            
            # Store the difference and the key
            differences.append((key, diff_tensor))
            
    # Sort differences in descending order and pick the top 1000 if required
    if top_1000:
        differences_new = sorted(differences, key=lambda x: x[1], reverse=True)[:1000]
    
    else:
        differences_new = differences

    # Aggregate differences by category
    for key, diff_tensor in differences_new:
        category = categorize_key(key, types[0])
        category_differences[category] += diff_tensor

    # Prepare data for the pie chart
    labels = list(category_differences.keys())
    sizes = list(category_differences.values())
    # Create the pie chart
    fig = plt.figure(figsize=(8, 8))
    plt.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%',  
        startangle=140,     
        colors=['orange','blue','green'],  
        shadow=False,        
        #explode=(0.1, 0.1, 0.1, 0.1)  
    )

    # Add a title
    plt.title(outputs[1], fontsize=14)

    plt.savefig(outputs[0], dpi=300, bbox_inches='tight')
    plt.close(fig)

def make_detailed_pie_chart(model1, model2, types, outputs):

    # Initialize a dictionary to hold total percentage differences for each category
    category_differences = defaultdict(float)
    stack_differences = defaultdict(float)  # To store total differences per stack
    node_edge_contributions = defaultdict(lambda: {"Node": 0, "Edge": 0})  # Node vs. Edge

    # Iterate through all keys in the models
    for key in model1.keys():
        if key in model2:  # Ensure the key exists in both models
            # Get parameters from both models
            param1 = model1[key].flatten().cpu().numpy()
            param2 = model2[key].flatten().cpu().numpy()
            
            if types[1] == "relative":
                diff_tensor = abs(param1 - param2).sum() / (abs(param1).sum() + 1e-8) * 100  # Avoid division by zero
            
            else:
                diff_tensor = abs(param1 - param2).sum() #absolute difference
            
            # Categorize the key and update the total percentage difference for the category
            category = categorize_key(key, types[0])
            category_differences[category] += diff_tensor
            
            # Categorize by stack and type
            if "_processor.gnn_stacks" in key:
                # Extract stack index
                stack_idx = int(key.split(".gnn_stacks.")[1].split(".")[0])
                stack_differences[stack_idx] += diff_tensor  # Add to the stack's total
                
                # Determine if it's Node or Edge
                if "node_fn" in key:
                    node_edge_contributions[stack_idx]["Node"] += diff_tensor
                elif "edge_fn" in key:
                    node_edge_contributions[stack_idx]["Edge"] += diff_tensor


    # Main Pie Chart (Top-Left)
    main_sizes = list(category_differences.values())
    main_labels = list(category_differences.keys())
    main_colors = ['orange', 'blue', 'green']

    # Convert data to lists for plotting
    stack_labels = [f"Stack {i}" for i in sorted(stack_differences.keys())]
    stack_sizes = [stack_differences[i] for i in sorted(stack_differences.keys())]
    node_edge_data = {f"Stack {i}": node_edge_contributions[i] for i in sorted(node_edge_contributions.keys())}

    # Plotting
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    ax_main = axs[0, 0]
    ax_main.pie(
        main_sizes,
        labels=main_labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=main_colors,
        shadow=False,
        #explode=(0.1, 0.1, 0.1, 0.1)
    )
    ax_main.set_title("Main Pie Chart")

    # Processor Breakdown Pie Chart (Top-Center)
    stack_colors = plt.cm.Paired(range(len(stack_labels)))
    ax_processor = axs[0, 2]
    ax_processor.pie(
        stack_sizes,
        labels=stack_labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=stack_colors,
        shadow=False
    )
    ax_processor.set_title("Processor Breakdown")

    # Add arrows from Processor to Processor Breakdown
    arrow = ConnectionPatch(
        xyA=(1.1, 0.2),  # Processor slice
        xyB=(-0.6, 0.2),  # Breakdown pie chart
        coordsA="data",
        coordsB="data",
        axesA=ax_main,
        axesB=ax_processor,
        arrowstyle="->",
        color="black"
    )
    fig.add_artist(arrow)

    # Node vs Edge Pie Charts for Each Stack
    for i, (stack, contributions) in enumerate(node_edge_data.items()):
        node, edge = contributions["Node"], contributions["Edge"]
        ax = axs[1 + i // 5, i % 5]  # Position on the grid
        ax.pie(
            [node, edge],
            labels=["Node", "Edge"],
            autopct='%1.1f%%',
            startangle=140,
            colors=['skyblue', 'lightcoral']
        )
        ax.set_title(f"{stack}: Node vs Edge")

    # Show the final plot
    plt.suptitle(outputs[1], fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.delaxes(axs[0,1])
    fig.delaxes(axs[0,3])
    fig.delaxes(axs[0,4])


    plt.savefig(outputs[0])

def plot_frequency_distribution(model1, model2, types, outputs, bins=50):
    """
    Plots a frequency distribution of the absolute parameter differences between two models.
    
    Args:
        model1 (dict): Parameters of the first model.
        model2 (dict): Parameters of the second model.
        output_path (str): Path to save the output plot.
        bins (int): Number of bins for the histogram.
    """
    
    all_differences = []

    # Iterate through all keys in model1
    for key in model1.keys():
        if key in model2:  # Ensure the key exists in both models
            # Flatten and convert parameters to numpy arrays
            param1 = model1[key].flatten().cpu().numpy()
            param2 = model2[key].flatten().cpu().numpy()

            if types[1] == "relative":
                diff_tensor = abs(param1 - param2).sum() / (abs(param1).sum() + 1e-8) * 100  # Avoid division by zero
            
            else:
                diff_tensor = abs(param1 - param2).sum() #absolute difference

            all_differences.extend(diff_tensor)

    # Convert list to a numpy array for plotting
    all_differences = np.array(all_differences)

    if types[1] == 'absolute':
        xlabel = "Absolute Parameter Differences"
    else:
        xlabel = "Relative Parameter Differences (%)"

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_differences, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(outputs[1], fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(outputs[0], dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Frequency distribution plot saved at {outputs[0]}")

def plot_cumulative_contribution(model1, model2, types, output_path):
    """
    Plots the cumulative contribution of the top parameters to the total absolute differences.

    Args:
        model1 (dict): Parameters of the first model.
        model2 (dict): Parameters of the second model.
        output_path (str): Path to save the output plot.
    """

    differences = []

    # Compute absolute differences for all matching keys
    for key in model1.keys():
        if key in model2:
            param1 = model1[key].flatten().cpu().numpy()
            param2 = model2[key].flatten().cpu().numpy()
            
            if types[1] == "relative":
                diff_tensor = np.abs(param1 - param2) / (np.abs(param1) + 1e-8) * 100  # Avoid division by zero
            
            else:
                diff_tensor = np.abs(param1 - param2) #absolute difference

            differences.extend(diff_tensor)

    # Sort absolute differences in descending order
    differences = np.array(differences)
    sorted_differences = np.sort(differences)[::-1]

    # Compute cumulative sum and normalize to percentage
    cumulative_sum = np.cumsum(sorted_differences)
    print(cumulative_sum[-1])
    cumulative_percentage = cumulative_sum / cumulative_sum[-1] * 100

    # Define the indices for top 10, 100, and 1000
    indices = [10, 100, 1000]

    # Plot cumulative contribution
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_percentage, label="Cumulative Contribution", linewidth=2)
    plt.scatter(indices, cumulative_percentage[indices], color="red", zorder=5, label="Top Parameters")
    
    # Add markers and annotations
    for idx in indices:
        plt.annotate(
            f"{cumulative_percentage[idx]:.1f}%",
            (idx, cumulative_percentage[idx]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="center",
            fontsize=12,
            color="black"
        )

    # Formatting the plot
    plt.xlabel("Number of Parameters", fontsize=14)
    plt.ylabel("Cumulative Contribution (%)", fontsize=14)
    plt.title("Cumulative Contribution of Parameters to Total Differences", fontsize=16)
    plt.axhline(100, color="gray", linestyle="--", alpha=0.7)  # Total contribution line
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.xscale('log')
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Cumulative contribution plot saved at {output_path}")

degrees = [35]
for deg in degrees:
    # Load the models
    model1 = torch.load("../ColumnCollaspeSimple/models/model.pt")
    #model1 = torch.load(f"../gns-reptile-task-encoder/models/vanilla_fine_tuned/except_decoder/models_{deg}_1/model-200.pt")
    model2 = torch.load("/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/models/reptile_training/models_8_test_5_all/model-600000.pt")

    outputs_bar_chart = ["../gns-reptile-task-encoder/plots/parameter_study/absolute_diff/top_parameters_models_8_test_5_all.png",
                        "Top 100 Parameters with changes for Reptile training with entire model"]

    outputs_pie_chart = [f"../gns-reptile-task-encoder/plots/parameter_study/vanilla_fine_tuned/encoder_and_processor/{deg}_deg/pie_chart_model-5000.png",
                        "Total Percentage Changes by Component for Reptile Training with Entire Model"]

    outputs_detailed_pie_chart = [f"../gns-reptile-task-encoder/plots/parameter_study/vanilla_fine_tuned/encoder_and_processor/{deg}_deg/pie_chart_detailed_model-5000.png",
                                "Breakdown of Relative Changes"]

    outputs_frequency_distribution = ["../gns-reptile-task-encoder/plots/parameter_study/absolute_diff/frequency_distribution_models_8_test_5_all.png",
                                    "Frequency Distribution of Absolute Parameter Differences"]

    output_cumulative_distribution = f"../gns-reptile-task-encoder/plots/parameter_study/relative_diff/cumulative_distribution_poster_model-5000.png"

    types = ["entire model",
            "relative"]

    # Extract parameter dictionaries
    params1 = model1['state_dict'] if 'state_dict' in model1 else model1
    params2 = model2['state_dict'] if 'state_dict' in model2 else model2

    # List to store percentage differences along with their keys and indices
    differences = []

    for key in params1.keys():
        if key in params2:
            param1 = params1[key]
            param2 = params2[key]
            
            # Ensure both parameters have the same shape
            if param1.shape == param2.shape:
                # Compute percentage difference tensor
                if types[1] == "relative":
                    diff_tensor = torch.abs(param1 - param2) / (torch.abs(param1) + 1e-8) * 100  # Avoid division by zero
                
                else:
                    diff_tensor = torch.abs(param1 - param2) #absolute difference
                
                # Flatten the tensor for easier processing
                flat_diffs = diff_tensor.view(-1).cpu().numpy()

                flat_param1 = param1.view(-1).cpu().numpy()
                flat_param2 = param2.view(-1).cpu().numpy()
                
                # Store differences with the key and index
                for idx, diff in enumerate(flat_diffs):
                    differences.append((key, idx, diff, flat_param1[idx], flat_param2[idx]))

    # Sort by percentage difference in descending order
    top_100 = sorted(differences, key=lambda x: x[2], reverse=True)[:100]

    # Print the top 1000 differences
    #for key, idx, diff, _, _ in top_100:
        #print(f"Key: {key}, Index: {idx}, Percentage Difference: {diff:.2f}%")

    #make_bar_chart(top_100, outputs_bar_chart, types[0])

    #make_pie_chart(model1, model2, outputs_pie_chart, types)

    #make_detailed_pie_chart(model1, model2, types, outputs_detailed_pie_chart)

    #plot_frequency_distribution(model1, model2, types, outputs_frequency_distribution)

    plot_cumulative_contribution(model1, model2, types, output_cumulative_distribution)