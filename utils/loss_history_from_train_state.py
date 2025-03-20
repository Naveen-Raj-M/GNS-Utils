import torch
import matplotlib.pyplot as plt

# Load the state dictionary
train_state_1 = torch.load("/scratch/10114/naveen_raj_manoharan/pretrained_model_1/models_2mp/model-999000.pt")
train_state_2 = torch.load("/scratch/10114/naveen_raj_manoharan/pretrained_model_1/models_10mp/train_state-409000.pt")


# Access loss history
train_loss_hist_1 = train_state_1['loss_history']['train']
#valid_loss_hist = train_state_1['loss_history']['valid']
train_loss_hist_2 = train_state_2['loss_history']['train']

epoch_1, train_loss_1 = zip(*train_loss_hist_1)
epoch_2, train_loss_2 = zip(*train_loss_hist_2)
#epoch, valid_loss = zip(*valid_loss_hist)

# Number of steps per epoch
steps_per_epoch = 5121

steps_1 = [epoch * steps_per_epoch for epoch in epoch_1]
steps_2 = [epoch * steps_per_epoch for epoch in epoch_2]

# Plotting train and valid loss histories
plt.plot(steps_1, train_loss_1, label='Train Loss with 2 message passing')
plt.plot(steps_2, train_loss_2, label='Train Loss with 10 message passing')
#plt.plot(steps, valid_loss, label='Validation Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.yscale('log')
plt.ylim(1e-3, 1)
plt.xlim()
plt.legend()
plt.title('Train vs Validation Loss')
plt.savefig("../pretrained_model_1/plots/loss_comparison_mp.png", dpi=300, bbox_inches='tight')

