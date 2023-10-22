"""

"""

import os, sys, time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import LandscapeSimulationDataset
from model import PhiNN
from helpers import select_device, jump_function, mean_cov_loss

# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--outdir', type=str, 
                    default="out/model_training")
parser.add_argument('--name', type=str, default="model")
parser.add_argument('-t', '--training_data', type=str, 
                    default="data/model_training_data")
parser.add_argument('-v', '--validation_data', type=str, 
                    default="data/model_validation_data")
parser.add_argument('-nt', '--nsims_training', type=int, default=100)
parser.add_argument('-nv', '--nsims_validation', type=int, default=30)
parser.add_argument('-nd', '--ndims', type=int, default=2)
parser.add_argument('-ns', '--nsigs', type=int, default=2)
parser.add_argument('-nc', '--ncells', type=int, default=100)
parser.add_argument('-dt', '--dt', type=float, default=1e-3)
parser.add_argument('--sigma', type=float, default=1e-3)

parser.add_argument('--use_gpu', action="store_true")
parser.add_argument('-e', '--num_epochs', type=int, default=50)
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--dtype', type=str, default="float32", 
                    choices=['float32', 'float64'])

parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

outdir = args.outdir
modelname = args.name
datdir_train = args.training_data
datdir_valid = args.validation_data
nsims_train = args.nsims_training
nsims_valid = args.nsims_validation
ndims = args.ndims
nsigs = args.nsigs
dt = args.dt
ncells = args.ncells
sigma = args.sigma
use_gpu = args.use_gpu
batch_size = args.batch_size
num_epochs = args.num_epochs
learning_rate = args.learning_rate
momentum = args.momentum
seed = args.seed
dtype = torch.float32 if args.dtype == 'float32' else torch.float64

time0 = time.time()

device = select_device() if use_gpu else 'cpu'
print(f"Using device: {device}")

rng = np.random.default_rng(seed=seed)
torch.manual_seed(int(rng.integers(100000, 2**32)))

train_dataset = LandscapeSimulationDataset(
    datdir_train, nsims_train, ndims, 
    transform='tensor', 
    target_transform='tensor',
    dtype=dtype,
)

validation_dataset = LandscapeSimulationDataset(
    datdir_valid, nsims_valid, ndims, 
    transform='tensor', 
    target_transform='tensor',
    dtype=dtype,
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
    ndim=ndims, nsig=nsigs, f_signal=f_signal, 
    ncells=ncells, 
    sigma=sigma,
    device=device,
    dtype=dtype,
).to(device)

outdir = "out/model_training"
os.makedirs(outdir, exist_ok=True)

loss_fn = mean_cov_loss
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=learning_rate, momentum=momentum)

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(train_dataloader):
        input, x1 = data

        # Zero gradients for the batch
        optimizer.zero_grad()

        # Evolve forward to get predicted state
        x1_pred = model(input.to(device), dt=dt)

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
            sys.stdout.flush()    
    return last_loss

#####################
##  Training Loop  ##
#####################

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

best_vloss = 1_000_000
for epoch in range(num_epochs):
    print(f'EPOCH {epoch + 1}:', flush=True)
    etime0 = time.time()
    
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch)

    # Empty cache
    torch.cuda.empty_cache()

    running_vloss = 0.0
    model.eval()

    for i, vdata in enumerate(validation_dataloader):
        vinputs, vx1 = vdata
        voutputs = model(vinputs.to(device))
        vloss = loss_fn(voutputs, vx1.to(device))
        running_vloss += vloss
    
    avg_vloss = running_vloss / (i + 1)
    print("LOSS [train: {}] [valid: {}] TIME [epoch: {:.3g} sec]".format(
        avg_loss, avg_vloss, time.time() - etime0), flush=True)

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f"{outdir}/{modelname}_{timestamp}_{epoch}"
        print("Saving model.")
        torch.save(model.state_dict(), model_path)
    
time1 = time.time()
print(f"Finished in {time1-time0:.3f} seconds.")
