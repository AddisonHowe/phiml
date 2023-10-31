"""Model training functions.

"""

import sys, time
import numpy as np
import torch
from datetime import datetime
from helpers import disp_mem_usage


def train_model(model, dt, loss_fn, optimizer, 
                train_dataloader, validation_dataloader, **kwargs):
    """Train a PhiNN model.

    Args:
        model (PhiNN): ...
        dt (float): ...
        loss_fn (callable): ...
        optimizer (callable): ...
        train_dataloader (DataLoader): ...
        validation_dataloader (DataLoader): ...
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    num_epochs = kwargs.get('num_epochs', 50)
    batch_size = kwargs.get('batch_size', 1)
    device = kwargs.get('device', 'cpu')
    model_name = kwargs.get('model_name', 'model')
    outdir = kwargs.get('outdir', 'out')
    verbosity = kwargs.get('verbosity', 1)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    np.savetxt(f"{outdir}/ncells.txt", [model.get_ncells()])
    np.savetxt(f"{outdir}/sigma.txt", [model.get_sigma()])

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    time0 = time.time()

    best_vloss = 1_000_000
    loss_hist_train = []
    loss_hist_valid = []
    for epoch in range(num_epochs):
        print(f'EPOCH {epoch + 1}:', flush=True)
        etime0 = time.time()
        
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        if verbosity >  1: disp_mem_usage("PRE TRAIN")
        print(model.get_sigma())
        avg_tloss = train_one_epoch(
            epoch, model, dt, loss_fn, optimizer, 
            train_dataloader,
            batch_size=batch_size,
            device=device,
            verbosity=verbosity,
        )
        if verbosity >  1: disp_mem_usage("POST TRAIN")

        running_vloss = 0.0
        model.eval()
        for i, vdata in enumerate(validation_dataloader):
            vinputs, vx1 = vdata
            voutputs = model(vinputs.to(device), dt=dt)
            vloss = loss_fn(voutputs, vx1.to(device))
            running_vloss += vloss.item()
            if verbosity >  1: disp_mem_usage(f"VALIDATION {i}")
        
        avg_vloss = running_vloss / (i + 1)
        print("LOSS [train: {}] [valid: {}] TIME [epoch: {:.3g} sec]".format(
            avg_tloss, avg_vloss, time.time() - etime0), flush=True)
        
        loss_hist_train.append(avg_tloss)
        loss_hist_valid.append(avg_vloss)
        np.save(f"{outdir}/training_loss_history.npy", loss_hist_train)
        np.save(f"{outdir}/validation_loss_history.npy", loss_hist_valid)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f"{outdir}/{model_name}_{timestamp}_{epoch}"
            print("Saving model.")
            torch.save(model.state_dict(), model_path)
        
    time1 = time.time()
    print(f"Finished training in {time1-time0:.3f} seconds.")


def train_one_epoch(epoch_idx, model, dt, loss_fn, optimizer,
                    dataloader, **kwargs):
    """One epoch of training.

    Args:
        epoch_idx (int): ...
        model (PhiNN): ...
        dt (float): ...
        loss_fn (callable): ...
        optimizer (callable): ...
        dataloader (DataLoader): ...

    Returns:
        float: last loss
    """
    #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
    batch_size = kwargs.get('batch_size', 1)
    device = kwargs.get('device', 'cpu')
    verbosity = kwargs.get('verbosity', 1)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(dataloader):  # train over batches
        input, x1 = data
        if verbosity >  1: disp_mem_usage('a')

        # Zero gradients for the batch
        optimizer.zero_grad()
        if verbosity >  1: disp_mem_usage('b')

        # Evolve forward to get predicted state
        x1_pred = model(input.to(device), dt=dt)
        if verbosity >  1: disp_mem_usage('c')

        if torch.any(torch.isnan(x1_pred)):
            print("Encountered nan")

        # Compute loss and its gradients
        loss = loss_fn(x1_pred, x1.to(device))
        if verbosity >  1: disp_mem_usage('d')
        loss.backward()
        if verbosity >  1: disp_mem_usage('e')

        # Adjust learning weights
        optimizer.step()
        if verbosity >  1: disp_mem_usage('f')

        # Gather data and report
        running_loss += loss.item()
        if i % batch_size == (batch_size-1):
            last_loss = running_loss / batch_size  # loss per batch
            print(f'\tbatch {i + 1} loss: {last_loss}', flush=True)
            running_loss = 0.

    return last_loss
