import os
import torch
#import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tensorboard_data(log_dir, tag):
    # Load TensorBoard logs
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get scalar data for the specified tag
    scalar_data = event_acc.Scalars(tag)
    steps = [x.step for x in scalar_data]
    values = [x.value for x in scalar_data]

    return steps, values


def make_subplot(plot_data, output_dir):
    
    n_tags = len(plot_data)
    n_cols = 3
    n_rows = (n_tags + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), squeeze=False)
    axes = axes.flatten()

    for idx, (tag, (steps, values)) in enumerate(plot_data.items()):
        ax = axes[idx]
        ax.plot(steps, values, label=tag, linewidth=1.5)
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1e-2)
        ax.grid(True)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(len(plot_data), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, "loss_history.png"), dpi=300, bbox_inches='tight')


def make_single_plot(tag, plot_data, output_dir):

    fig,ax = plt.subplots(1, 1, figsize=(15, 12))
    steps, values = plot_data[tag]

    ax.plot(steps, values, label=tag, marker="o", linewidth=1.5)
    ax.set_title(tag, fontsize=10)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1e-2)
    ax.grid(True)
    ax.legend(fontsize=8)

    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, f"loss_history_{tag}.png"), dpi=300, bbox_inches='tight')

def make_multi_plot(labels, title, plot_data, output_dir):

    fig,ax = plt.subplots(1, 1, figsize=(15, 12))

    for label in labels:
        steps, values = plot_data[label]
        ax.plot(steps, values, label=label, linewidth=1.5)
    
    ax.set_title(title, fontsize=25)
    ax.set_xlabel("Steps", fontsize=18)
    ax.set_ylabel("Loss", fontsize=18)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1)
    ax.grid(True)
    ax.legend(fontsize=18)

    plt.tight_layout()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(os.path.join(output_dir, f"loss_history_comaprison_mp.png"), dpi=300, bbox_inches='tight')


log_dir_1 = "/scratch/10114/naveen_raj_manoharan/logs/pretrained_model_1_2mp"
log_dir_2 = "/scratch/10114/naveen_raj_manoharan/logs/pretrained_model_1_10mp"
output_dir = "../pretrained_model_1/plots/"

'''tags = ["Loss/train", 
        "Loss/task_0", 
        "Loss/task_1", 
        "Loss/task_2",  
        "Loss/valid_0", 
        "Loss/valid_1", 
        "Loss/valid_2",
        "Loss/train_iter",
        "Loss/valid_iter"]'''

tags = ["Loss/train_epoch"]
plot_data = {}
labels = [
    "2_message_passing",
    "10_message_passing"
]
title = "Comparison of GNS Training Loss with 2 and 10 Message Passing Layers"

for tag in tags:
    try:
        steps_1, values_1 = extract_tensorboard_data(log_dir_1, tag)
        steps_2, values_2 = extract_tensorboard_data(log_dir_2, tag)
        #steps = np.concatenate((steps_1, steps_2))
        #values = np.concatenate((steps_1, steps_2))
        #plot_data[tag] = (steps_1, values_1)
        #print(type(plot_data[tag]))
        print(labels[0])
        plot_data[labels[0]] = (steps_1, values_1)
        plot_data[labels[1]] = (steps_2, values_2)
    except KeyError:
        print(f"Tag {tag} not found in the logs.")

# make_single_plot(tags[0], plot_data, output_dir)
#make_subplot(plot_data, output_dir)
make_multi_plot(labels, title, plot_data, output_dir)

