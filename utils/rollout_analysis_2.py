import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def read_rollout_data(rollout_path: str, rollout_filename: str) -> dict:
    """
    Reads rollout data from a pickle file.

    Args:
        rollout_path (str): Path to the directory containing the rollout file.
        rollout_filename (str): Name of the rollout file (without extension).

    Returns:
        dict: Loaded rollout data.
    """
    file_path = os.path.join(rollout_path, f"{rollout_filename}.pkl")
    with open(file_path, "rb") as file:
        return pickle.load(file)


def get_trajectories(rollout_data: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts MPM and predicted trajectories from rollout data.

    Args:
        rollout_data (dict): Rollout data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: MPM and predicted trajectories.
    """
    predicted_trajectory = np.concatenate([
        rollout_data["initial_positions"],
        rollout_data["predicted_rollout"]
    ], axis=0)
    mpm_trajectory = np.concatenate([
        rollout_data["initial_positions"],
        rollout_data["ground_truth_rollout"]
    ], axis=0)
    return mpm_trajectory, predicted_trajectory

def get_normalized_parameters(trajectories: tuple, rollout_data) -> dict:

    normalized_parameters = {
        "mpm": {}, 
        "gns": {}
        }
    
    keys = list(normalized_parameters.keys())

    timesteps = rollout_data["metadata"]["sequence_length"]
    dt = rollout_data["metadata"]["dt"]
    g = 9.81 * dt ** 2


    for i, trajectory in enumerate(trajectories):
        
        # find displacements
        displacements = []
        initial_position = trajectory[0, :]
       
        for current_position in trajectory:
            displacement = initial_position - current_position
            scalr_disp = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
            displacements.append(scalr_disp)
        
        displacements = np.array(displacements)

        # find velocities
        velocities = trajectory[1:, ] - trajectory[:-1, ]  
        initial_velocity = velocities[0]  
        initial_velocity = np.expand_dims(initial_velocity, 0)  
        velocities = np.concatenate((velocities, initial_velocity))  
        scalar_velocities = np.sqrt(velocities[:, :, 0] ** 2 + velocities[:, :, 1] ** 2)

        # find runout length and height
        L_t = []
        H_t = []
        
        for position in trajectory:
            L = np.percentile(position[:, 0], output_percentile) - np.min(trajectory[0][:, 0])  # front end of runout
            H = np.percentile(position[:, 1], output_percentile) - np.min(trajectory[0][:, 1])  # top of runout
            L_t.append(L)
            H_t.append(H)
        
        # normalize H, L with initial length of column and time with critical time
        L_initial = np.amax(trajectory[0][:, 0]) - np.min(trajectory[0][:, 0])
        H_initial = np.amax(trajectory[0][:, 1]) - np.min(trajectory[0][:, 1])
        critical_time = np.sqrt(H_initial / g)
        time = np.arange(0, timesteps)  # assume dt=1
        normalized_time = time / critical_time  # critical time assuming dt=1
        normalized_L_t = (L_t - L_initial) / L_initial
        normalized_H_t = H_t / L_initial

         # compute energies
        potentialE = np.sum(mass * g * trajectory[:, :, 1], axis=-1)  # sum(mass * gravity * elevation)
        kineticE = (1 / 2) * np.sum(mass * scalar_velocities ** 2, axis=-1)
        E0 = potentialE[0] + kineticE[0]
        dissipationE = E0 - kineticE - potentialE
        
        # normalize energies
        normalized_Ek = kineticE / E0
        normalized_Ep = potentialE / E0
        normalized_Ed = dissipationE / E0

        normalized_parameters[keys[i]] = {
            "normalized_time": normalized_time,
            "normalized_runout": normalized_L_t,
            "normalized_height": normalized_H_t,
            "normalized_kinetic_energy": normalized_Ek,
            "normalized_potential_energy": normalized_Ep,
            "normalized_dissipation_energy": normalized_Ed 
        }

    return normalized_parameters

def calculate_output_timesteps(height: float, g: float, timesteps: int) -> List[int]:
    """
    Calculate the output timesteps for plotting.

    Args:
        height (float): Initial height of the trajectory.
        g (float): Gravitational constant.
        timesteps (int): Total number of timesteps.

    Returns:
        List[int]: List of output timesteps.
    """
    critical_time = np.sqrt(height / g)
    output_critical_timesteps = [0, 2.5, 4.0, timesteps / critical_time]
    output_timesteps = np.around(np.array(output_critical_timesteps) * critical_time, 0).astype(int).tolist()
    output_timesteps[-1] = timesteps - 1
    return output_timesteps


def plot_runout_and_height(normalized_parameters_list: dict, legends: Tuple[List[str], List[str]],
                            colors: List[str], lines: List[str]) -> plt.Figure:
    """
    Plot runout and height.

    Args:
        normalized_parameters (dictionary): Normalized parameters for plots
        legends (Tuple[List[str], List[str]]): Legends for the plots.
        colors (List[str]): Colors for the plots.
        lines (List[str]): Line styles for the plots.

    Returns:
        plt.Figure: The plot figure.
    """
    fig, ax1 = plt.subplots(figsize=(12, 10))
    ax2 = ax1.twinx()

    for i in range(4):
        '''ax1.plot(normalized_parameters_list[i]["mpm"]["normalized_time"], 
                normalized_parameters_list[i]["mpm"]["normalized_runout"], 
                color=colors[2*i], 
                linestyle=lines[0], 
                label=legends[2*i][0])
        
        ax2.plot(normalized_parameters_list[i]["mpm"]["normalized_time"], 
                normalized_parameters_list[i]["mpm"]["normalized_height"], 
                color=colors[2*i], 
                linestyle=lines[1], 
                label=legends[(2*i)+1][0])'''

        ax1.plot(normalized_parameters_list[i]["gns"]["normalized_time"], 
                normalized_parameters_list[i]["gns"]["normalized_runout"], 
                color=colors[(2*i)+1], 
                linestyle=lines[0], 
                label=legends[2*i][1])
        
        ax2.plot(normalized_parameters_list[i]["gns"]["normalized_time"], 
                normalized_parameters_list[i]["gns"]["normalized_height"], 
                color=colors[(2*i)+1], 
                linestyle=lines[1], 
                label=legends[(2*i)+1][1])

    ax1.set_xlabel(r"$t / \tau_{c}$", fontsize=18)
    ax1.set_ylabel(r"$(L_t - L_0)/L_0$", fontsize=18)
    ax2.set_ylabel(r"$H_t/L_0$", fontsize=18)

    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 3.5)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 2.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, ncol=2, loc="upper right", prop={'size': 15})

    fig.tight_layout()
    return fig


def plot_energy(normalized_parameters: dict,
                colors: List[List[str]], lines: List[str], legends: Tuple[List[str], List[str], List[str]]) -> plt.Figure:
    """
    Plot normalized energies.

    Args:
        normalized_parameters (dictionary): Normalized parameters for the plots.
        colors (List[List[str]]): Colors for the plots.
        lines (List[str]): Line styles for the plots.
        legends (Tuple[List[str], List[str], List[str]]): Legends for the plots.

    Returns:
        plt.Figure: The plot figure.
    """
    fig, ax1 = plt.subplots(figsize=(12, 12))
    ax2 = ax1.twinx()

    sample_rate = 1
    ax1.plot(normalized_parameters["mpm"]["normalized_time"][::sample_rate], 
             normalized_parameters["mpm"]["normalized_potential_energy"][::sample_rate],
             color=colors[0][0], 
             linestyle=lines[0], 
             label=legends[0][0])
    
    ax2.plot(normalized_parameters["mpm"]["normalized_time"][::sample_rate], 
            normalized_parameters["mpm"]["normalized_kinetic_energy"][::sample_rate],
            color=colors[1][0], 
            linestyle=lines[1], 
            label=legends[1][0])
    
    ax2.plot(normalized_parameters["mpm"]["normalized_time"][::sample_rate], 
             normalized_parameters["mpm"]["normalized_dissipation_energy"][::sample_rate],
             color=colors[2][0], 
             linestyle=lines[2], 
             label=legends[2][0])
    
    ax1.plot(normalized_parameters["gns"]["normalized_time"][::sample_rate], 
             normalized_parameters["gns"]["normalized_potential_energy"][::sample_rate],
             color=colors[0][1], 
             linestyle=lines[0], 
             label=legends[0][1])
    
    ax2.plot(normalized_parameters["gns"]["normalized_time"][::sample_rate], 
            normalized_parameters["gns"]["normalized_kinetic_energy"][::sample_rate],
            color=colors[1][1], 
            linestyle=lines[1], 
            label=legends[1][1])
    
    ax2.plot(normalized_parameters["gns"]["normalized_time"][::sample_rate], 
             normalized_parameters["gns"]["normalized_dissipation_energy"][::sample_rate],
             color=colors[2][1], 
             linestyle=lines[2], 
             label=legends[2][1])

    ax1.set_xlabel(r"$t / \tau_{c}$", fontsize=18)
    ax1.set_ylabel(r"$E_p/E_0$", fontsize=18)
    ax2.set_ylabel(r"$E_k/E_0$ and $E_d/E_0$", fontsize=18)

    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    ax1.set_xlim(0, 4)
    ax1.set_ylim(0)
    ax2.set_ylim(0, 0.7)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2,
                          ncol=2, loc="upper right", prop={'size': 15})

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    rollout_path = "../test/"
    rollout_filenames = ["20_deg_vanilla",
                         "25_deg_vanilla",
                         "35_deg_vanilla",
                         "40_deg_vanilla",
    ]
    output_percentile = 99.9
    output_filename = "different_phi"
    mass = 1

    rollout_data_list = []
    trajectories_list = []
    normalized_parameters_list = []

    for i in range(4):
        rollout_data = read_rollout_data(rollout_path, rollout_filenames[i])
        trajectories = get_trajectories(rollout_data)
        normalized_parameters = get_normalized_parameters(trajectories, rollout_data)

        rollout_data_list.append(rollout_data)
        trajectories_list.append(trajectories)
        normalized_parameters_list.append(normalized_parameters)


    #height = np.max(trajectories_1[0][0][:, 1])

    #boundary = rollout_data["metadata"]["bounds"]
    #timesteps = rollout_data["metadata"]["sequence_length"]
    #dt = rollout_data["metadata"]["dt"]
    #g = 9.81 * dt ** 2

    #output_timesteps = calculate_output_timesteps(height, g, timesteps)

    runout_legends = (
        ["MPM Runout 20_deg", "GNS Runout 20_deg"],
        ["MPM Height 20_deg", "GNS Height 20_deg"],
        ["MPM Runout 25_deg", "GNS Runout 25_deg"],
        ["MPM Height 25_deg", "GNS Height 25_deg"],
        ["MPM Runout 35_deg", "GNS Runout 35_deg"],
        ["MPM Height 35_deg", "GNS Height 35_deg"],
        ["MPM Runout 40_deg", "GNS Runout 40_deg"],
        ["MPM Height 40_deg", "GNS Height 40_deg"])
    runout_lines = ["solid", "dashed"]
    runout_colors = ["silver", "black",
                     "red", "orange",
                     "blue", "skyblue",
                     "lime", "green"]

    fig = plot_runout_and_height(normalized_parameters_list,
                           runout_legends,
                           runout_colors,
                           runout_lines)
    
    fig.savefig(os.path.join(rollout_path, f"{output_filename}_vanilla_only_runout.png"))

    '''energy_legends = (
        ["MPM $E_p/E_0$", "GNS $E_p/E_0$"],
        ["MPM $E_k/E_0$", "GNS $E_k/E_0$"],
        ["MPM $E_d/E_0$", "GNS $E_d/E_0$"])
    energy_lines = ["solid", "dashed", "dotted"]
    energy_colors = [["silver", "black"], ["lightcoral", "darkred"], ["lightsteelblue", "darkblue"]]

    fig = plot_energy(normalized_parameters,
                           energy_colors,
                           energy_lines,
                           energy_legends)
    
    fig.savefig(os.path.join(rollout_path, f"{output_filename}_energy.png"))'''


