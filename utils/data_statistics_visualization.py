import numpy as np
from matplotlib import pyplot as plt
import os

def load_trajectories(filepath: str) -> list[np.ndarray]:
    """
    Load trajectories from a .npz file.

    Args:
        filepath (str): Path to the .npz file containing trajectories.

    Returns:
        list[np.ndarray]: List of trajectories loaded from the file.
    """
    with np.load(filepath, allow_pickle=True) as data_file:
        data = [item for _, item in data_file.items()]
    return data

def calculate_velocity(positions: np.ndarray, dt: float = 1) -> np.ndarray:
    """
    Calculate velocity using finite differences.

    Args:
        positions (np.ndarray): Positions array of shape (nsteps, nparticles, ndim)
        dt (float): Time step interval

    Returns:
        np.ndarray: Velocity array of shape (nsteps, nparticles, ndim)
    """
    velocity = np.diff(positions, axis=0) / dt
    
    # Add a zero velocity at the start of the trajectory to make the shape consistent with the positions
    velocities = np.concatenate([np.zeros((1, positions.shape[1], positions.shape[2])), velocity], axis=0)
    
    return velocities

def calculate_acceleration(velocity: np.ndarray, dt: float = 1) -> np.ndarray:
    """
    Calculate acceleration using finite differences.

    Args:
        velocity (np.ndarray): Velocity array of shape (nsteps, nparticles, ndim)
        dt (float): Time step interval

    Returns:
        np.ndarray: Acceleration array of shape (nsteps, nparticles, ndim)
    """
    acceleration = np.diff(velocity, axis=0) / dt

    # Add a zero acceleration at the start of the trajectory to make the shape consistent with the positions
    accelerations = np.concatenate([np.zeros((1, velocity.shape[1], velocity.shape[2])), acceleration], axis=0)
    
    return accelerations

def normalize(data: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    """
    Normalize the data using the provided mean and standard deviation.

    Args:
        data (np.ndarray): Data array to be normalized.
        mean (list[float]): Mean values.
        std (list[float]): Standard deviation values.

    Returns:
        np.ndarray: Normalized data array.
    """
    return (data - np.array(mean)) / np.array(std)

def process_trajectories(filepath: str, mean_vel: list[float], std_vel: list[float], mean_acc: list[float], std_acc: list[float]) -> dict[str, dict[str, np.ndarray]]:
    """
    Process trajectories by calculating velocities, accelerations, and their normalized versions.

    Args:
        filepath (str): Path to the .npz file containing trajectories.
        mean_vel (list[float]): Mean values for velocity normalization.
        std_vel (list[float]): Standard deviation values for velocity normalization.
        mean_acc (list[float]): Mean values for acceleration normalization.
        std_acc (list[float]): Standard deviation values for acceleration normalization.

    Returns:
        dict[str, dict[str, np.ndarray]]: Dictionary containing processed trajectories with positions, velocities, accelerations, 
                                          normalized velocities, and normalized accelerations.
    """
    # Load the trajectories from the file
    data = load_trajectories(filepath)

    # Initialize an empty dictionary to store processed trajectories
    trajectories = {}

    for i, trajectory in enumerate(data):
        if i in range(3, 6):
            continue
        # Extract positions and calculate velocities and accelerations from the trajectory
        positions = trajectory[0]
        velocities = calculate_velocity(positions)
        accelerations = calculate_acceleration(velocities)
        
        # Normalize velocities and accelerations
        normalized_velocities = normalize(velocities, mean_vel, std_vel)
        normalized_accelerations = normalize(accelerations, mean_acc, std_acc)
        
        # Store the processed data in the dictionary
        trajectories[f'trajectory_{i}'] = {
            'positions': positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'normalized_velocities': normalized_velocities,
            'normalized_accelerations': normalized_accelerations
        }

    return trajectories

def scatter_plot(data: np.ndarray, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    """
    Create a scatter plot and save it to a file.

    Args:
        data (np.ndarray): Data array of shape (n, 2) to be plotted.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        filename (str): Filename to save the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_statistics(filepath: str, trajectories: dict[str, dict[str, np.ndarray]]) -> None:
    """
    Plot statistics of velocities and accelerations before and after normalization.

    Args:
        trajectories (dict[str, dict[str, np.ndarray]]): Dictionary containing processed trajectories with positions, velocities, 
                                                         accelerations, normalized velocities, and normalized accelerations.

    Returns:
        None
    """
    # Flatten the data for scatter plot
    all_velocities = np.concatenate([traj['velocities'] for traj in trajectories.values()], axis=0)
    all_accelerations = np.concatenate([traj['accelerations'] for traj in trajectories.values()], axis=0)
    all_normalized_velocities = np.concatenate([traj['normalized_velocities'] for traj in trajectories.values()], axis=0)
    all_normalized_accelerations = np.concatenate([traj['normalized_accelerations'] for traj in trajectories.values()], axis=0)

    # Reshape the data to 2D arrays for plotting
    all_velocities_flat = all_velocities.reshape(-1, all_velocities.shape[-1])
    all_accelerations_flat = all_accelerations.reshape(-1, all_accelerations.shape[-1])
    all_normalized_velocities_flat = all_normalized_velocities.reshape(-1, all_normalized_velocities.shape[-1])
    all_normalized_accelerations_flat = all_normalized_accelerations.reshape(-1, all_normalized_accelerations.shape[-1])

    os.makedirs(filepath, exist_ok=True)
    # Scatter plots before normalization
    filename = os.path.join(filepath, 'velocity_before_normalization.png')
    scatter_plot(all_velocities_flat, 'Velocity Before Normalization', 'Velocity X', 'Velocity Y', filename)
    filename = os.path.join(filepath, 'acceleration_before_normalization.png')
    scatter_plot(all_accelerations_flat, 'Acceleration Before Normalization', 'Acceleration X', 'Acceleration Y', filename)

    # Scatter plots after normalization
    filename = os.path.join(filepath, 'normalized_velocity.png')
    scatter_plot(all_normalized_velocities_flat, 'Normalized Velocity', 'Velocity X', 'Velocity Y', filename)
    filename = os.path.join(filepath, 'normalized_acceleration.png')
    scatter_plot(all_normalized_accelerations_flat, 'Normalized Acceleration', 'Acceleration X', 'Acceleration Y', filename)

def main() -> None:
    """
    Main function to process trajectories and plot statistics.

    Args:
        None

    Returns:
        None
    """
    # Filepath to the .npz file containing trajectories
    filepath: str = '../gns-reptile/datasets/reptile_training/train.npz'
    savepath: str = '../gns-reptile/plots/reptile-dataset/old_statistics'
    
    '''# Mean and standard deviation values for normalization (reptile dataset)
    mean_vel: list[float] = [-2.34548054e-05, -1.66995219e-04]
    std_vel: list[float] = [0.00043458, 0.00037027]
    mean_acc: list[float] = [-9.65125001e-10, -1.15959126e-10]
    std_acc: list[float] = [1.36493267e-05, 1.59062523e-05]'''

    # Mean and standard deviation values for normalization (pre-trained dataset)
    mean_vel: list[float] = [-8.534464569480504e-08, -0.0005064359707880266]
    std_vel: list[float] = [0.001487010805936108, 0.0015684952903373933]
    mean_acc: list[float] = [-2.022114849552206e-09, -4.190964926618496e-09]
    std_acc: list[float] = [0.00016380678562445672, 0.00017673744575316674]

    # Process the trajectories
    trajectories: dict[str, dict[str, np.ndarray]] = process_trajectories(filepath, mean_vel, std_vel, mean_acc, std_acc)
    
    # Plot statistics of the processed trajectories
    plot_statistics(savepath, trajectories)

if __name__ == '__main__':
    main()