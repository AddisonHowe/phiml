"""Main script.

"""

import os, sys
import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import LandscapeSimulationDataset
from model import PhiNN
from model_training import train_model
from helpers import select_device, jump_function, mean_cov_loss, kl_divergence_est


def parse_args(args):
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

    parser.add_argument('--infer_noise', action="store_true")
    parser.add_argument('--sigma', type=float, default=1e-3)

    parser.add_argument('--use_gpu', action="store_true")
    parser.add_argument('-e', '--num_epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--optimizer', type=str, default="sgd", 
                        choices=['sgd', 'adam', 'rms'])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--dtype', type=str, default="float32", 
                        choices=['float32', 'float64'])

    parser.add_argument('--continuation', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--no_timestamp', action="store_false")

    return parser.parse_args(args)


def main(args):
    outdir = args.outdir
    model_name = args.name
    datdir_train = args.training_data
    datdir_valid = args.validation_data
    nsims_train = args.nsims_training
    nsims_valid = args.nsims_validation
    ndims = args.ndims
    nsigs = args.nsigs
    dt = args.dt
    ncells = args.ncells
    infer_noise = args.infer_noise
    sigma = args.sigma
    use_gpu = args.use_gpu
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    optimization_method = args.optimizer
    learning_rate = args.learning_rate
    momentum = args.momentum
    continuation_fpath = args.continuation
    seed = args.seed
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64

    device = select_device() if use_gpu else 'cpu'
    print(f"Using device: {device}")

    if not seed:
        seed = np.random.randint(2**32)
    print(f"Using seed: {seed}")

    if continuation_fpath:
        print(f"Continuing training of model {continuation_fpath}")
    
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(int(rng.integers(100000, 2**32)))

    if not args.no_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = outdir + "_" + timestamp

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

    # Construct the model
    f_signal = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
    model = PhiNN(
        ndim=ndims, nsig=nsigs, 
        f_signal=f_signal, nsigparams=5,
        ncells=ncells, 
        infer_noise=infer_noise,
        sigma=sigma,
        device=device,
        dtype=dtype,
        sample_cells=True,
        rng=rng,
    ).to(device)

    if continuation_fpath:
        model.load_state_dict(
            torch.load(continuation_fpath, map_location=torch.device(device))
        )

    os.makedirs(outdir, exist_ok=True)

    # loss_fn = mean_cov_loss
    loss_fn = kl_divergence_est

    optimizer = select_optimizer(model, optimization_method, args)

    train_model(
        model, dt, loss_fn, optimizer, 
        train_dataloader, validation_dataloader,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        model_name=model_name,
        outdir=outdir,
    )

def select_optimizer(model, optimization_method, args):
    if optimization_method == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate, 
            momentum=args.momentum,
        )
    elif optimization_method == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate,
        )
    elif optimization_method == 'rms':
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=args.learning_rate,
            momentum=args.momentum,
        )
    else:
        msg = f"{optimization_method} optimization not implemented."
        raise NotImplementedError(msg)
    return optimizer

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    print(args)
    main(args)
