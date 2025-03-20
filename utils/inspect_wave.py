import numpy as np
import sys
import pickle

def inspect_npz(file_path):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            # Extract particle positions at the first time step
                first_time_step_positions = data["trajectory0"][0][3]  # Shape: (nparticles, ndim)
                second_time_step_positions = data["trajectory0"][0][4]

                 # Filter positions based on third dimension condition
                mask = (first_time_step_positions[:, 2] < 0.25) | (first_time_step_positions[:, 2] > 1.5)
                filtered_first_positions = first_time_step_positions[mask]
                filtered_second_positions = second_time_step_positions[mask]

                
                if filtered_first_positions.size > 0:
                    min_values = np.min(filtered_first_positions, axis=0)
                    max_values = np.max(filtered_first_positions, axis=0)
                    
                    print(f"Min values along each dimension (filtered): {min_values}")
                    print(f"Max values along each dimension (filtered): {max_values}")
                    
                    # Compute average velocity between first and second time step
                    velocities = filtered_second_positions - filtered_first_positions
                    avg_velocity = np.mean(velocities, axis=0)
                    print(f"Average velocity along each dimension: {avg_velocity/0.0025}")
                
                else:
                    print("No positions satisfy the filtering condition.")
             
    except Exception as e:
        print(f"Error loading NPZ file: {e}")

if __name__ == "__main__":
    
    filepath = "/scratch/10114/naveen_raj_manoharan/Agentic_Text2Sim/trajectory0_car_wave.npz"
    inspect_npz(filepath)


