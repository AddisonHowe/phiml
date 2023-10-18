"""

"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import LandscapeSimulationDataset
from model import PhiNN
from helpers import select_device, jump_function, mean_cov_loss

# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

SEED = 0
OUTDIR = f"out/model_training"
TRAINING_DATA = "data/model_training_data"
VALIDATION_DATA = "data/model_validation_data"
NSIMS_TRAINING = 5
NSIMS_VALIDATION = 2
NDIM = 2
NSIGS = 2
DT = 1e-3
NCELLS = 100
SIGMA=1e-3

USE_GPU = True
batch_size = 8

device = select_device() if USE_GPU else 'cpu'
print(f"Using device: {device}")

rng = np.random.default_rng(seed=SEED)
torch.manual_seed(int(rng.integers(100000, 2**32)))

train_dataset = LandscapeSimulationDataset(
    TRAINING_DATA, NSIMS_TRAINING, NDIM, 
    transform='tensor', 
    target_transform='tensor'
)

validation_dataset = LandscapeSimulationDataset(
    VALIDATION_DATA, NSIMS_VALIDATION, NDIM, 
    transform='tensor', 
    target_transform='tensor'
)

train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)

validation_dataloader = DataLoader(validation_dataset, 
                                   batch_size=batch_size, 
                                   shuffle=False)

# Construct the model
f_signal = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
model = PhiNN(
    ndim=NDIM, nsig=NSIGS, f_signal=f_signal, 
    ncells=NCELLS, 
    sigma=SIGMA,
    device=device,
).to(device)

outdir = "out/model_training"
os.makedirs(outdir, exist_ok=True)

LEARNING_RATE = 1e-3
MOMENTUM = 0.9


loss_fn = mean_cov_loss
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=LEARNING_RATE, momentum=MOMENTUM)

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(train_dataloader):
        input, x1 = data

        # Zero gradients for the batch
        optimizer.zero_grad()

        # Evolve forward to get predicted state
        x1_pred = model(input.to(device), dt=DT)

        # Compute loss and its gradients
        loss = loss_fn(x1_pred, x1.to(device))
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % batch_size == (batch_size-1):
            last_loss = running_loss / batch_size  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

#####################
##  Training Loop  ##
#####################

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

NUM_EPOCHS = 3
best_vloss = 1_000_000
for epoch in range(NUM_EPOCHS):
    print(f'EPOCH {epoch + 1}:')
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch)

    running_vloss = 0.0
    model.eval()

    for i, vdata in enumerate(validation_dataloader):
        vinputs, vx1 = vdata
        voutputs = model(vinputs.to(device))
        vloss = loss_fn(voutputs, vx1.to(device))
        running_vloss += vloss
    
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"{OUTDIR}/model_{timestamp}_{epoch}"
        torch.save(model.state_dict(), model_path)
