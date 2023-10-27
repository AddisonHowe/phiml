import pytest
import os, glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import PhiNN
from model_training import train_model
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


TRAINDIR = "tests/simtest2/data_train"
VALIDDIR = "tests/simtest2/data_valid"
NSIMS_TRAIN = 4
NSIMS_VALID = 4

OUTDIR = "tests/simtest2/tmp_out"

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################


@pytest.mark.parametrize('device, dtype', [
    ['cpu', torch.float32], 
    ['cpu', torch.float64], 
    # ['mps', torch.float32], # ERRORS because of bug in pytorch?
])
@pytest.mark.parametrize('batch_size, batch_sims, final_ws_exp', [
    [2, [['sim0', 'sim1'], ['sim2', 'sim3']],  # train over 2 batches
     [[[1.02157,2.991],[1.9923,1.9557],[3.02391,0.994084]],  # W1 final expected
      [[0.876646,0.892136,-2.11692],[-0.102348,0.913638,-0.0664326],[-1.01595,1.98658,0.986411]],  # W2 final expected
      [[2.00485,2.96534,0.959294]],  # W3 final expected
      [[2.06403,3.95404],[-0.991957,0.962239]],  # WT final expected
    ]],  
])
class TestTraining:

    def _load_data(self, dtype, batch_size):
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
        return train_dataset, train_dataloader, valid_dataset, valid_dataloader

    def _get_model(self, device, dtype):
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
        return model
    
    def _remove_files(self, outdir, name):
        # Remove generated files
        for filename in glob.glob(f"{outdir}/{name}*"):
            os.remove(filename) 
    
    def test_1_epoch_train_2_batch(self, device, dtype, 
                                      batch_size, batch_sims, final_ws_exp):
        learning_rate = 0.1
        dt = 0.1
        loss_fn = mean_diff_loss
        errors1 = []

        _, train_dloader, _, valid_dloader = self._load_data(dtype, batch_size)
        
        model = self._get_model(device, dtype)
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
        )

        oldparams = [p.detach().numpy().copy() for p in model.parameters()]

        train_model(
            model, dt, loss_fn, optimizer,
            train_dloader, valid_dloader,
            num_epochs=1,
            batch_size=batch_size,
            device=device,
            outdir=OUTDIR,
            model_name='tmp_model',
        )

        self._remove_files(OUTDIR, 'tmp_model')

        newparams = [p.detach().numpy().copy() for p in model.parameters()]

        if len(newparams) != 4:
            msg = "Bad length for parameters after training 1 epoch. " + \
                f"Expected 4. Got {len(newparams)}."
            errors1.append(msg)
        
        for i in range(4):
            oldparam_act = oldparams[i]
            newparam_act = newparams[i]
            newparam_exp = final_ws_exp[i]
            if not np.allclose(newparam_exp, newparam_act):
                msg = f"Error in w{i}:\nExpected:\n{newparam_exp}\nGot:\n{newparam_act}"
                errors1.append(msg)

        assert not errors1, \
            "Errors occurred in epoch 1:\n{}".format("\n".join(errors1))
        