"""Benchmarking Tests

pytest -s --benchmark tests/benchmark.py

"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.profiler import profiler, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from model import PhiNN
from dataset import LandscapeSimulationDataset
from helpers import jump_function, mean_cov_loss, kl_divergence_est

TRAINDIR = "data/benchmark_data_train"
VALIDDIR = "data/benchmark_data_valid"
NSIMS_TRAIN = 20
NSIMS_VALID = 5

def get_data_loaders(dtype, batch_size):
    train_dataset = LandscapeSimulationDataset(
        TRAINDIR, NSIMS_TRAIN, 2, 
        transform='tensor', 
        target_transform='tensor',
        dtype=dtype,
    )

    validation_dataset = LandscapeSimulationDataset(
        VALIDDIR, NSIMS_VALID, 2, 
        transform='tensor', 
        target_transform='tensor',
        dtype=dtype,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )

    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    return train_dataloader, validation_dataloader

OUTDIR = "tests/benchmark/tmp_out"

@pytest.mark.benchmark
@pytest.mark.parametrize("device", ['cuda'])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("infer_noise", [True])
class TestBenchmarkModel:

    def _get_model(self, device, dtype, infer_noise):
        f_signal = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
        model = PhiNN(
            ndim=2, nsig=2, 
            f_signal=f_signal, nsigparams=5,
            ncells=100, 
            infer_noise=infer_noise,
            sigma=1e-3,
            device=device,
            dtype=dtype,
            sample_cells=True,
        ).to(device)
        return model

    @pytest.mark.parametrize('batch_size', [2, 4, 8])
    @pytest.mark.parametrize('loss_fn', [mean_cov_loss, kl_divergence_est])
    def test_benchmark_forward(self, device, dtype, infer_noise, 
                               batch_size, loss_fn):
        model = self._get_model(device, dtype, infer_noise)
        train_loader, _ = get_data_loaders(dtype, batch_size)
        inputs, x1 = next(iter(train_loader))
        inputs = inputs.to(device)
        x1 = x1.to(device)
        dt = 1e-1
        model(inputs, dt=dt)  # warm-up
        with profiler.profile(activities=[ProfilerActivity.CPU], 
                              profile_memory=False,) as prof:
            with record_function("forward"):
                prediction = model(inputs, dt=dt)
            with record_function("loss"):    
                loss = loss_fn(prediction, x1)
            with record_function("backward"):
                loss.backward()
    
        print(prof.key_averages(group_by_stack_n=5).table(
              sort_by='self_cpu_time_total', row_limit=10))
