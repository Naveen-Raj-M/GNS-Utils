import numpy as np

input_path = '../terramechanics/datasets/train.npz'

with np.load(input_path, allow_pickle=True) as data_file:
    data = [item for _, item in data_file.items()]

dt = 1

# Initialize empty lists to store data
velocity = []
acceleration = []
num_data = []
# Calculate velocity and acceleration statistics for each trajectory independently
for trajectory in data:
    positions = trajectory[0]
    
    # Calculate velocity as the difference in position between consecutive time steps, divided by dt
    velocities = np.diff(positions, axis=0) / dt  # Shape: (ntimestep - 1, n_particles, ndim)
    # Add a zero velocity at the start of the trajectory to make the shape consistent with the positions
    velocities = np.concatenate([np.zeros((1, velocities.shape[1], velocities.shape[2])), velocities], axis=0)
    
    # Calculate acceleration as the difference in velocity between consecutive time steps, divided by dt
    accelerations = np.diff(velocities, axis=0) / dt  # Shape: (ntimestep - 2, n_particles, ndim)
    # Add a zero acceleration at the start of the trajectory to make the shape consistent with the positions
    accelerations = np.concatenate([np.zeros((1, accelerations.shape[1], accelerations.shape[2])), accelerations], axis=0)

    # Store results
    velocity.append(velocities)
    acceleration.append(accelerations)
    num_data.append(velocities.shape[0] * velocities.shape[1])  # timesteps * particles

# compute the mean
velocity_mean = np.sum([np.sum(velocity[i],axis=(0, 1)) for i in range(len(velocity))], axis=0) / np.sum(num_data)
acceleration_mean = np.sum([np.sum(acceleration[i],axis=(0, 1)) for i in range(len(acceleration))], axis=0) / np.sum(num_data)

# compute the std dev
velocity_std = np.sqrt(np.sum([np.sum((velocity[i] - velocity_mean)**2, axis=(0,1)) for i in range(len(velocity))], axis=0) / np.sum(num_data))
acceleration_std = np.sqrt(np.sum([np.sum((acceleration[i] - acceleration_mean)**2, axis=(0,1)) for i in range(len(acceleration))], axis=0) / np.sum(num_data))


print("Velocity Mean:", velocity_mean)
print("Velocity Std Dev:", velocity_std)
print("Acceleration Mean:", acceleration_mean)
print("Acceleration Std Dev:", acceleration_std)