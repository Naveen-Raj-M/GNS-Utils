import numpy as np
import os

data = []

input_path = "../terramechanics/datasets/npz_files/"
#input_path = "../ColumnCollapseSimple/datasets/rollouts"
#input_path = "../Agentic_Text2Sim/Sim_Inputs/test_old.npz"
output_path = f"../terramechanics/datasets/"

phi = [30]

all_trajectories = {}
counter = 0

'''for angle in phi:
    for i in range(2):
        for j in range(3):
            file_path = os.path.join(input_path, f"{angle}_{i}_{j}", "trajectory0.npz")
            data = np.load(file_path, allow_pickle=True)

            positions = data["trajectory0"][0]
            particle_types = data["trajectory0"][1]

            # tan_friction = np.tan(np.radians(angle))
            # tan_friction_array = np.full_like(particle_types, tan_friction, dtype=np.float64)

            all_trajectories[f"trajectory_{counter}"] = np.array([positions, particle_types], dtype=object)
            # all_trajectories[f"trajectory_0"] = np.array([positions, particle_types], dtype=object) #new

            # output_file_path = os.path.join(output_path, f"test-{counter}.npz") #new
            #np.savez_compressed(output_file_path, **all_trajectories) #new
            # print(f"Trajectory saved to {output_path}") #new
            
            counter += 1
            data.close()'''


for angle in phi:
    for i in range(30):
        #file_path = os.path.join(input_path, f"{angle}_deg_{i}.npz")
        file_path = os.path.join(input_path, f"{angle}_deg_{i}.npz")
        
        if not os.path.exists(file_path):
            print(f"warning: {file_path} does not exist")
            continue

        data = np.load(file_path, allow_pickle=True)
        key = data.files[0]
        print(key)
        positions = data[key][0]
        particle_types = data[key][1]
        tan_friction = np.tan(np.radians(angle))
        tan_friction_array = np.full_like(particle_types, tan_friction, dtype=np.float64)

        all_trajectories[f"trajectory_{counter}"] = np.array([positions, particle_types], dtype=object)

        counter += 1
        data.close()
    output_path_2 = os.path.join(output_path, "test.npz")
    np.savez_compressed(output_path_2, **all_trajectories)
    print(f"All trajectories saved to {output_path_2}")
'''
with np.load(input_path, allow_pickle=True) as data_file:
    data = [item for _, item in data_file.items()]

for trajectory in data:
    positions = trajectory[0]
    particle_types = trajectory[1]
    particle_types[particle_types == 6] = 5

    tan_friction = np.tan(np.radians(15))
    tan_friction_array = np.full_like(particle_types, tan_friction, dtype=np.float64)
    all_trajectories[f"trajectory_{counter}"] = np.array([positions, particle_types], dtype=object)
    counter += 1'''

#np.savez_compressed(output_path, **all_trajectories)
#print(f"All trajectories saved to {output_path}")