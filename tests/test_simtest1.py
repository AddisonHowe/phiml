import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import PhiNN
from main import train_model
from dataset import LandscapeSimulationDataset
from helpers import get_binary_function
from helpers import jump_function, mean_cov_loss, mean_diff_loss

#####################
##  Configuration  ##
#####################

W1 = np.array([
    [1, 3],
    [2, 2],
    [3, 1],
], dtype=float)

W2 = np.array([
    [1, 1, -2],
    [0, 1, 0],
    [-1, 2, 1],
], dtype=float)

W3 = np.array([
    [2, 3, 1]
], dtype=float)

WT1 = np.array([
    [2, 4],
    [-1, 1],
], dtype=float)


TRAINDIR = "tests/simtest1/data_train"
VALIDDIR = "tests/simtest1/data_valid"
NSIMS_TRAIN = 4
NSIMS_VALID = 4

OUTDIR = "tests/simtest1/tmp_out"

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch_size', [1, 2])
def test_dataloader(device, dtype, batch_size):
    train_dataset = LandscapeSimulationDataset(
        TRAINDIR, NSIMS_TRAIN, 2, 
        transform='tensor', 
        target_transform='tensor',
        dtype=dtype,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )


@pytest.mark.parametrize('device, dtype', [
    ['cpu', torch.float32], 
    ['cpu', torch.float64], 
    # ['mps', torch.float32], # ERRORS because of bug in pytorch?
])
@pytest.mark.parametrize('batch_size, batch_sims', [
    [4, [['sim0', 'sim1', 'sim2', 'sim3']]],
    # [2, [['sim0', 'sim1'], ['sim2', 'sim3']]],
    # [1, [['sim0'], ['sim1'], ['sim2'], ['sim3']]],
])
def test_1_epoch_train_full_batch(device, dtype, batch_size, batch_sims):

    train_dataset = LandscapeSimulationDataset(
        TRAINDIR, NSIMS_TRAIN, 2, 
        transform='tensor', 
        target_transform='tensor',
        dtype=dtype,
    )
    valid_dataset = LandscapeSimulationDataset(
        VALIDDIR, NSIMS_VALID, 2, 
        transform='tensor', 
        target_transform='tensor',
        dtype=dtype,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )

    # Construct the model
    f_signal = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
    model = PhiNN(
        ndim=2, nsig=2, f_signal=f_signal, 
        ncells=4, sigma=0,
        device=device,
        dtype=dtype,
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi=[W1, W2, W3],
        init_weight_values_tilt=[WT1],
        testing_dw=torch.tensor([[0.,0.],[0.,0.],[0.,0.],[0.,0.]], 
                                dtype=dtype),
    ).to(device)

    oldparams = [p.detach().numpy().copy() for p in model.parameters()]

    loss_fn = mean_diff_loss

    learning_rate=0.1
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
    )

    dt = 1e-1
    train_model(
        model, dt, loss_fn, optimizer,
        train_dataloader, valid_dataloader,
        num_epochs=1,
        batch_size=batch_size,
        device=device,
        outdir=OUTDIR,
    )
    
    newparams = [p.detach().numpy().copy() for p in model.parameters()]

    # Compute the expected new parameter values by hand
    batch_dl_dw1 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw1.npy') 
                              for s in b] for b in batch_sims])
    batch_dl_dw2 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw2.npy') 
                              for s in b] for b in batch_sims])
    batch_dl_dw3 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw3.npy') 
                              for s in b] for b in batch_sims])
    batch_dl_dwt = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dwt.npy') 
                              for s in b] for b in batch_sims])
    
    batch_dl_dw1_avg = np.mean(batch_dl_dw1, axis=1)
    batch_dl_dw2_avg = np.mean(batch_dl_dw2, axis=1)
    batch_dl_dw3_avg = np.mean(batch_dl_dw3, axis=1)
    batch_dl_dwt_avg = np.mean(batch_dl_dwt, axis=1)

    batch_grad_avgs = [batch_dl_dw1_avg, batch_dl_dw2_avg, 
                       batch_dl_dw3_avg, batch_dl_dwt_avg]
    
    errors = []
    for i in range(4):
        oldparam_act = oldparams[i]
        newparam_act = newparams[i]
        print(newparam_act)
        new_param_exp = oldparam_act - learning_rate * batch_grad_avgs[i]
        if not np.allclose(new_param_exp, newparam_act):
            msg = f"Error in w{i}:\nExpected:\n{new_param_exp}\nGot:\n{newparam_act}"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

@pytest.mark.parametrize('device, dtype', [
    ['cpu', torch.float32], 
    ['cpu', torch.float64], 
    # ['mps', torch.float32], # ERRORS because of bug in pytorch?
])
@pytest.mark.parametrize('batch_size, batch_sims', [
    [4, [['sim0', 'sim1', 'sim2', 'sim3']]],
    # [2, [['sim0', 'sim1'], ['sim2', 'sim3']]],
    # [1, [['sim0'], ['sim1'], ['sim2'], ['sim3']]],
])
def test_2_epoch_train_full_batch(device, dtype, batch_size, batch_sims):

    train_dataset = LandscapeSimulationDataset(
        TRAINDIR, NSIMS_TRAIN, 2, 
        transform='tensor', 
        target_transform='tensor',
        dtype=dtype,
    )
    valid_dataset = LandscapeSimulationDataset(
        VALIDDIR, NSIMS_VALID, 2, 
        transform='tensor', 
        target_transform='tensor',
        dtype=dtype,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )

    # Construct the model
    f_signal = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
    model = PhiNN(
        ndim=2, nsig=2, f_signal=f_signal, 
        ncells=4, sigma=0,
        device=device,
        dtype=dtype,
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi=[W1, W2, W3],
        init_weight_values_tilt=[WT1],
        testing_dw=torch.tensor([[0.,0.],[0.,0.],[0.,0.],[0.,0.]], 
                                dtype=dtype),
    ).to(device)

    oldparams = [p.detach().numpy().copy() for p in model.parameters()]

    loss_fn = mean_diff_loss

    learning_rate=0.1
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=learning_rate, 
    )

    dt = 1e-1
    train_model(
        model, dt, loss_fn, optimizer,
        train_dataloader, valid_dataloader,
        num_epochs=1,
        batch_size=batch_size,
        device=device,
        outdir=OUTDIR,
    )
    
    newparams = [p.detach().numpy().copy() for p in model.parameters()]

    # Compute the expected new parameter values by hand
    batch_dl_dw1 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw1.npy') 
                              for s in b] for b in batch_sims])
    batch_dl_dw2 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw2.npy') 
                              for s in b] for b in batch_sims])
    batch_dl_dw3 = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dw3.npy') 
                              for s in b] for b in batch_sims])
    batch_dl_dwt = np.array([[np.load(f'{TRAINDIR}/{s}/dloss_dwt.npy') 
                              for s in b] for b in batch_sims])
    
    batch_dl_dw1_avg = np.mean(batch_dl_dw1, axis=1)
    batch_dl_dw2_avg = np.mean(batch_dl_dw2, axis=1)
    batch_dl_dw3_avg = np.mean(batch_dl_dw3, axis=1)
    batch_dl_dwt_avg = np.mean(batch_dl_dwt, axis=1)

    batch_grad_avgs = [batch_dl_dw1_avg, batch_dl_dw2_avg, 
                       batch_dl_dw3_avg, batch_dl_dwt_avg]
    
    errors1 = []
    params_after_1_epoch = []
    for i in range(4):
        oldparam_act = oldparams[i]
        newparam_act = newparams[i]
        new_param_exp = oldparam_act - learning_rate * batch_grad_avgs[i]
        params_after_1_epoch.append(new_param_exp)
        if not np.allclose(new_param_exp, newparam_act):
            msg = f"Error in w{i}:\nExpected:\n{new_param_exp}\nGot:\n{newparam_act}"
            errors1.append(msg)

    # Create new model to train for 2 epochs
    model2 = PhiNN(
        ndim=2, nsig=2, f_signal=f_signal, 
        ncells=4, sigma=0,
        device=device,
        dtype=dtype,
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi=[W1, W2, W3],
        init_weight_values_tilt=[WT1],
        testing_dw=torch.tensor([[0.,0.],[0.,0.],[0.,0.],[0.,0.]], 
                                dtype=dtype),
    ).to(device)

    optimizer = torch.optim.SGD(
        model2.parameters(), 
        lr=learning_rate, 
    )

    train_model(
        model2, dt, loss_fn, optimizer,
        train_dataloader, valid_dataloader,
        num_epochs=2,
        batch_size=batch_size,
        device=device,
        outdir=OUTDIR,
    )

    newparams = [p.detach().numpy().copy() for p in model2.parameters()]

    # Compute the expected new parameter values by hand
    batch_dl_dw1 = np.array([[np.load(f'{TRAINDIR}/{s}/e2_dloss_dw1.npy') 
                              for s in b] for b in batch_sims])
    batch_dl_dw2 = np.array([[np.load(f'{TRAINDIR}/{s}/e2_dloss_dw2.npy') 
                              for s in b] for b in batch_sims])
    batch_dl_dw3 = np.array([[np.load(f'{TRAINDIR}/{s}/e2_dloss_dw3.npy') 
                              for s in b] for b in batch_sims])
    batch_dl_dwt = np.array([[np.load(f'{TRAINDIR}/{s}/e2_dloss_dwt.npy') 
                              for s in b] for b in batch_sims])
    
    batch_dl_dw1_avg = np.mean(batch_dl_dw1, axis=1)
    batch_dl_dw2_avg = np.mean(batch_dl_dw2, axis=1)
    batch_dl_dw3_avg = np.mean(batch_dl_dw3, axis=1)
    batch_dl_dwt_avg = np.mean(batch_dl_dwt, axis=1)

    batch_grad_avgs = [batch_dl_dw1_avg, batch_dl_dw2_avg, 
                       batch_dl_dw3_avg, batch_dl_dwt_avg]
    
    errors2 = []
    for i in range(4):
        oldparam_act = params_after_1_epoch[i]
        newparam_act = newparams[i]
        new_param_exp = oldparam_act - learning_rate * batch_grad_avgs[i]
        if not np.allclose(new_param_exp, newparam_act):
            msg = f"Error in w{i}:\nExpected:\n{new_param_exp}\nGot:\n{newparam_act}"
            errors2.append(msg)
    
    assert not errors1, "Errors occurred in epoch 1:\n{}".format("\n".join(errors1))
    assert not errors2, "Errors occurred in epoch 2:\n{}".format("\n".join(errors2))



