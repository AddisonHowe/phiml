"""Main script.

"""

import os, sys
import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from phiml.dataset import LandscapeSimulationDataset
from phiml.model import PhiNN
from phiml.model_training import train_model
from phiml.helpers import select_device, jump_function, mean_cov_loss, kl_divergence_est


def parse_args(args):
    parser = argparse.ArgumentParser()
    # Training options
    parser.add_argument('--name', type=str, default="model")
    parser.add_argument('-o', '--outdir', type=str, 
                        default="out/model_training")
    parser.add_argument('-t', '--training_data', type=str, 
                        default="data/model_training_data")
    parser.add_argument('-v', '--validation_data', type=str, 
                        default="data/model_validation_data")
    parser.add_argument('-nt', '--nsims_training', type=int, default=100)
    parser.add_argument('-nv', '--nsims_validation', type=int, default=30)
    parser.add_argument('--use_gpu', action="store_true")
    parser.add_argument('-e', '--num_epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=32)

    # Model options
    parser.add_argument('-nd', '--ndims', type=int, default=2,
                        help="Number of state space dimensions for the data.")
    parser.add_argument('-ns', '--nsigs', type=int, default=2, 
                        help="Number of signals in the system.")
    parser.add_argument('-nc', '--ncells', type=int, default=100, 
                        help="Number of cells to evolve internally.")
    parser.add_argument('-dt', '--dt', type=float, default=1e-3,
                        help="Euler timestep to use internally.")
    parser.add_argument('--signal_function', type=str, default='jump',
                        choices=['jump'], 
                        help="Identifier for the signal function.")

    parser.add_argument('--hidden_dims', type=int, nargs='+', 
                        default=[16, 32, 32, 16])
    parser.add_argument('--hidden_acts', type=str, nargs='+', 
                        default=4*['softplus'])
    parser.add_argument('--final_act', type=str, default='softplus')
    parser.add_argument('--layer_normalize', default=False)

    parser.add_argument('--infer_noise', action="store_true",
                        help="If specified, infer the noise level.")
    parser.add_argument('--sigma', type=float, default=1e-3,
                        help="Noise level if not inferring sigma." + \
                        "Otherwise, the initial value for the sigma parameter.")

    # Loss function options
    parser.add_argument('--loss', type=str, default="kl", 
                        choices=['kl', 'mcd'], 
                        help='kl: KL divergence est; mcd: mean+cov difference.')
    parser.add_argument('--continuation', type=str, default=None, 
                        help="Path to file with model parameters to load.")
    
    # Optimizer options
    parser.add_argument('--optimizer', type=str, default="sgd", 
                        choices=['sgd', 'adam', 'rms'])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.)
    
    parser.add_argument('--dtype', type=str, default="float32", 
                        choices=['float32', 'float64'])
    
    # Misc. options
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timestamp', action="store_true",
                        help="Add timestamp to out directory.")

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
    hidden_dims = args.hidden_dims
    hidden_acts = args.hidden_acts
    final_act = args.final_act
    layer_normalize = args.layer_normalize
    infer_noise = args.infer_noise
    sigma = args.sigma
    use_gpu = args.use_gpu
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    optimization_method = args.optimizer
    learning_rate = args.learning_rate
    momentum = args.momentum
    weight_decay = args.weight_decay
    cont_path = args.continuation
    loss_fn_key = args.loss
    signal_function_key = args.signal_function
    seed = args.seed
    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    do_plot = args.plot

    device = select_device() if use_gpu else 'cpu'
    print(f"Using device: {device}")
    
    seed = seed if seed else np.random.randint(2**32)
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(int(rng.integers(2**32)))
    print(f"Using seed: {seed}")

    if cont_path: 
        print(f"Continuing training of model {cont_path}")
    
    if args.timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir = outdir + "_" + timestamp

    # Get training and validation dataloaders
    train_dataloader, valid_dataloader = get_dataloaders(
        datdir_train, datdir_valid, nsims_train, nsims_valid, 
        batch_size_train=batch_size, batch_size_valid=batch_size, 
        ndims=ndims, dtype=dtype, return_datasets=False
    )

    # Construct the model
    f_signal, nsigparams = get_signal_function(signal_function_key)
    model = PhiNN(
        ndim=ndims, nsig=nsigs, 
        f_signal=f_signal, nsigparams=nsigparams,
        ncells=ncells, 
        hidden_dims=hidden_dims,
        hidden_acts=hidden_acts,
        final_act=final_act,
        layer_normalize=layer_normalize,
        infer_noise=infer_noise,
        sigma=sigma,
        device=device,
        dtype=dtype,
        sample_cells=True,
        rng=rng,
    ).to(device, dtype=dtype)

    if cont_path:
        model.load_state_dict(
            torch.load(cont_path, map_location=torch.device(device))
        )

    loss_fn = select_loss_function(loss_fn_key)
    optimizer = select_optimizer(model, optimization_method, args)
    
    os.makedirs(outdir, exist_ok=True)
    if do_plot:
        os.makedirs(f"{outdir}/images", exist_ok=True)

    log_args(outdir, args)
    log_model(outdir, model)

    train_model(
        model, dt, loss_fn, optimizer, 
        train_dataloader, valid_dataloader,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        model_name=model_name,
        outdir=outdir,
        plotting=do_plot,
        plotting_opts={},  # Default plotting options
    )

def get_dataloaders(datdir_train, datdir_valid, nsims_train, nsims_valid, 
                    batch_size_train=1, batch_size_valid=1, 
                    ndims=2, dtype=None, return_datasets=False):
    train_dataset = LandscapeSimulationDataset(
        datdir_train, nsims_train, ndims, 
        transform='tensor', 
        target_transform='tensor',
        dtype=dtype,
    )

    valid_dataset = LandscapeSimulationDataset(
        datdir_valid, nsims_valid, ndims, 
        transform='tensor', 
        target_transform='tensor',
        dtype=dtype,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size_train, 
        shuffle=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=batch_size_valid, 
        shuffle=False,
    )
    if return_datasets:
        return train_dataloader, valid_dataloader, train_dataset, valid_dataset
    else:
        return train_dataloader, valid_dataloader
    
def get_signal_function(key):
    if key == 'jump':
        f_signal = lambda t,p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
        nsigparams = 5
    else:
        msg = f"Unknown signal function identifier {key}."
        raise RuntimeError(msg)
    return f_signal, nsigparams

def select_loss_function(key):
    if key == 'kl':
        return kl_divergence_est
    elif key == 'mcd':
        return mean_cov_loss
    else:
        msg = f"Unknown loss function identifier {key}."
        raise RuntimeError(msg)

def select_optimizer(model, optimization_method, args):
    if optimization_method == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate, 
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif optimization_method == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    elif optimization_method == 'rms':
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        msg = f"{optimization_method} optimization not implemented."
        raise RuntimeError(msg)
    return optimizer

def log_args(outdir, args):
    with open(f"{outdir}/log_args.txt", 'w') as f:
        for arg, value in sorted(vars(args).items()):
            f.write(str(arg) + " : " + "%r" % value + "\n")

def log_model(outdir, model):
    np.savetxt(f"{outdir}/ncells.txt", [model.get_ncells()])
    np.savetxt(f"{outdir}/sigma.txt", [model.get_sigma()])
    with open(f"{outdir}/log_model.txt", 'w') as f:
        for arg, value in sorted(vars(model).items()):
            f.write(str(arg) + " : " + "%r" % value + "\n")

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    print('args:', args)
    main(args)
