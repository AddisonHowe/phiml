"""Model training functions.

"""

import sys, time
import torch
from datetime import datetime


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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    time0 = time.time()

    best_vloss = 1_000_000
    for epoch in range(num_epochs):
        print(f'EPOCH {epoch + 1}:', flush=True)
        etime0 = time.time()
        
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            epoch, model, dt, loss_fn, optimizer, 
            train_dataloader,
            batch_size=batch_size,
            device=device,
        )

        # Empty cache
        # torch.cuda.empty_cache()

        running_vloss = 0.0
        model.eval()

        for i, vdata in enumerate(validation_dataloader):
            vinputs, vx1 = vdata
            voutputs = model(vinputs.to(device), dt=dt)
            vloss = loss_fn(voutputs, vx1.to(device))
            running_vloss += vloss
        
        avg_vloss = running_vloss / (i + 1)
        print("LOSS [train: {}] [valid: {}] TIME [epoch: {:.3g} sec]".format(
            avg_loss, avg_vloss, time.time() - etime0), flush=True)

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f"{outdir}/{model_name}_{timestamp}_{epoch}"
            print("Saving model.")
            torch.save(model.state_dict(), model_path)
        
    time1 = time.time()
    print(f"Finished in {time1-time0:.3f} seconds.")


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
    for i, data in enumerate(dataloader):
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
