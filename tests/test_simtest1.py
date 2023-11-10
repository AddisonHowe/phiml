import pytest
import os, glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from phiml.model import PhiNN
from phiml.model_training import train_model
from phiml.dataset import LandscapeSimulationDataset
from phiml.helpers import get_binary_function
from phiml.helpers import jump_function, mean_cov_loss, mean_diff_loss

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


@pytest.mark.parametrize('device, dtype, atol', [
    ['cpu', torch.float32, 1e-4], 
    ['cpu', torch.float64, 1e-6], 
    ['cuda', torch.float32, 1e-4], 
    ['cuda', torch.float64, 1e-6], 
    ['mps', torch.float32, 1e-4],
    # ['mps', torch.float64, 1e-6],  # mps does not support float64
])
@pytest.mark.parametrize('batch_size, batch_sims', [
    [4, [['sim0', 'sim1', 'sim2', 'sim3']]],  # train over all 4 in one batch
])
class TestTwoStepSimulation:

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
            testing=True, include_signal_bias=False, 
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
    
    def test_1_epoch_train_full_batch(self, device, dtype, atol, 
                                      batch_size, batch_sims):
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip("cuda not available")
        if device == 'mps' and 'mps' not in dir(torch.backends):
            pytest.skip("mps not available")
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

        oldparams = [p.detach().cpu().numpy().copy() for p in model.parameters()]

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

        newparams = [p.detach().cpu().numpy().copy() for p in model.parameters()]

        if len(newparams) != 4:
            msg = "Bad length for parameters after training 1 epoch. " + \
                f"Expected 4. Got {len(newparams)}."
            errors1.append(msg)

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
        
        params_after_1_epoch = []  # store parameters after 1 epoch training
        for i in range(4):
            oldparam_act = oldparams[i]
            newparam_act = newparams[i]
            newparam_exp = oldparam_act - learning_rate * batch_grad_avgs[i]
            params_after_1_epoch.append(newparam_exp)
            if not np.allclose(newparam_exp, newparam_act, atol=atol):
                msg = f"Error in w{i}:\nExpected:\n{newparam_exp}\nGot:\n{newparam_act}"
                errors1.append(msg)

        assert not errors1, \
            "Errors occurred in epoch 1:\n{}".format("\n".join(errors1))
        

    def test_2_epoch_train_full_batch(self, device, dtype, atol, 
                                      batch_size, batch_sims):
        if device == 'cuda' and not torch.cuda.is_available():
            pytest.skip("cuda not available")
        if device == 'mps' and 'mps' not in dir(torch.backends):
            pytest.skip("mps not available")
        learning_rate = 0.1
        dt = 0.1
        loss_fn = mean_diff_loss
        errors1 = []
        errors2 = []
        
        _, train_dloader, _, valid_dloader = self._load_data(dtype, batch_size)
        
        model = self._get_model(device, dtype)
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
        )

        oldparams = [p.detach().cpu().numpy().copy() for p in model.parameters()]

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
        
        newparams = [p.detach().cpu().numpy().copy() for p in model.parameters()]

        if len(newparams) != 4:
            msg = "Bad length for parameters after training 1 epoch. " + \
                f"Expected 4. Got {len(newparams)}."
            errors1.append(msg)

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
        
        params_after_1_epoch = []  # store parameters after 1 epoch training
        for i in range(4):
            oldparam_act = oldparams[i]
            newparam_act = newparams[i]
            newparam_exp = oldparam_act - learning_rate * batch_grad_avgs[i]
            params_after_1_epoch.append(newparam_exp)
            if not np.allclose(newparam_exp, newparam_act, atol=atol):
                msg = f"Error in w{i}:\nExpected:\n{newparam_exp}\nGot:\n{newparam_act}"
                errors1.append(msg)
        
        # Create new model to train for 2 epochs
        model2 = self._get_model(device, dtype)
        optimizer = torch.optim.SGD(
            model2.parameters(), 
            lr=learning_rate, 
        )

        train_model(
            model2, dt, loss_fn, optimizer,
            train_dloader, valid_dloader,
            num_epochs=2,
            batch_size=batch_size,
            device=device,
            outdir=OUTDIR,
            model_name='tmp_model',
        )

        self._remove_files(OUTDIR, 'tmp_model')

        newparams = [p.detach().cpu().numpy().copy() for p in model2.parameters()]
        
        if len(newparams) != 4:
            msg = "Bad length for parameters after training 2 epochs. " + \
                f"Expected 4. Got {len(newparams)}."
            errors2.append(msg)

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
        
        for i in range(4):
            oldparam_act = params_after_1_epoch[i]
            newparam_act = newparams[i]
            new_param_exp = oldparam_act - learning_rate * batch_grad_avgs[i]
            if not np.allclose(new_param_exp, newparam_act, atol=atol):
                msg = f"Error in w{i}:\nExpected:\n{new_param_exp}\nGot:\n{newparam_act}"
                errors2.append(msg)

        assert not errors1, \
            "Errors occurred in epoch 1:\n{}".format("\n".join(errors1))
        assert not errors2, \
            "Errors occurred in epoch 2:\n{}".format("\n".join(errors2))
        