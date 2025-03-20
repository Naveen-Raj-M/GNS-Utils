import numpy as np
import os

#path = "../gns-reptile-task-encoder/datasets/reptile_training_4/train_0.npz"
#path = "../gns-reptile/column_collapse_taichi_2/datasets/train_1.npz"
path="../terramechanics/datasets/npz_files/"

for i in [29]:
    if os.path.exists(os.path.join(path,f"30_deg_{i}.npz")):
        with np.load(os.path.join(path,f"30_deg_{i}.npz"), allow_pickle=True) as data_file:
            # Print the keys in the dataset
            print("Keys in the dataset:")
            for key in data_file.keys():
                print(key)

            data = [item for _, item in data_file.items()]

            print(len(data))
            print(np.shape(data[0]))
            print(np.shape(data[0][0]))
            print(data[0][1])
    
    else:
        pass
# Replace 6 with 5 in data[0][1]
#data[0][1][data[0][1] == 6] = 5

#print(data[0][1])
#output_path = path="../Agentic_Text2Sim/Sim_Inputs/test.npz"
#np.savez_compressed(output_path, data)
#print(np.shape(data[0][2]))
#print(data[0][2][0])

'''print(np.shape(data[0][2]))
print(data[0][2])
print(data[5][2])'''


'''train_data = data[:7]
data_dict = {f"simulation_trajectory_{i}": train_data[i] for i in range(7)}
np.savez('../column_collapse_30/datasets/valid.npz', **data_dict)'''




'''ranges = [
    (0, 5),
    (6, 11),
    (12, 17)
]

# Split the list into the specified ranges
split_data = [data[start:end+1] for start, end in ranges]

# Store each segment into a dictionary and save as separate .npz files
for i, sublist in enumerate(split_data):
    data_dict = {f"simulation_trajectory_{j}": sublist[j] for j in range(len(sublist))}
    np.savez(f"../column_collapse_taichi_2/datasets/train_{i}.npz", **data_dict)'''