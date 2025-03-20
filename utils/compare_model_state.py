import torch

def load_state_dict(path):
    """Load the state dictionary from a .pt file."""
    return torch.load(path, map_location=torch.device("cpu"))

def compare_state_dicts(state_dict1, state_dict2):
    """Compare two state dictionaries and print the keys of the parameters that have been updated."""
    updated_keys = []
    for key in state_dict1.keys():
        if key in state_dict2:
            if not torch.equal(state_dict1[key], state_dict2[key]):
                updated_keys.append(key)
        else:
            print(f"Key {key} is missing in the second state dictionary.")
    
    return updated_keys

# Load the state dictionaries from the two .pt files
path1 = "/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/models/reptile_training/on_partially_trained/with_extra_dimension/model-500000/models_2_test_p/model-0.pt"
# path2 = "../ColumnCollapseSimple/models/model.pt"
path2 = "/scratch/10114/naveen_raj_manoharan/gns-reptile-task-encoder/models/reptile_training/on_partially_trained/with_extra_dimension/model-500000/models_2_test_p/model-176.pt"

state_dict1 = load_state_dict(path1)
state_dict2 = load_state_dict(path2)

# Compare the state dictionaries and get the updated keys
updated_keys = compare_state_dicts(state_dict1, state_dict2)

# Print the updated keys
print("Updated keys:")
for key in updated_keys:
    print(key)

'''checkpoint = torch.load(path1)

# Print all the keys in the checkpoint dictionary
print("Keys in the checkpoint file:")
for key in checkpoint.keys():
    print(key)

for key in checkpoint["global_train_state"].keys():
    print(key)
print(type(checkpoint["optimizer_state"]))
#print(type(checkpoint["loss_history"]["train"]))
#print(type(checkpoint["loss_history"]["valid"]))
#print(checkpoint["global_train_state"]["epoch"])'''