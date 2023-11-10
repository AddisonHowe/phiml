import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from .model import PhiNN
from .helpers import jump_function

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-d', '--modeldir', type=str, default="models")
parser.add_argument('-o', '--outdir', type=str, default="out/model_analysis")
parser.add_argument('-v', '--verbosity', type=int, default=1)
args = parser.parse_args()

model = args.model
modeldir = f"{args.modeldir}/{model}"
outdir = f"{args.outdir}/{model}"
verbosity = args.verbosity

os.makedirs(outdir, exist_ok=True)

ARGS_TO_LOAD = {
    'ndims', 'nsigs', 'ncells', 'hidden_dims', 'hidden_acts', 'final_act', 
    'layer_normalize', 'infer_noise', 'sigma', 'dtype', 
    'training_data', 'validation_data', 'nsims_training', 'nsims_validation',
}

def load_args_from_log(logfilepath, args_to_load=ARGS_TO_LOAD):
    args = {}
    with open(logfilepath, 'r') as f:
        for line in f.readlines():
            line = line[0:-1]  # remove \n
            line = line.split(" : ")  # try to split into key, val pair
            if len(line) == 2:
                key, val = line
                if key in args_to_load:
                    args[key] = eval(val)
    return args

def load_model_directory(modeldir, modelname, verbosity=0):
    loss_hist_train = np.load(f"{modeldir}/training_loss_history.npy")
    loss_hist_valid = np.load(f"{modeldir}/validation_loss_history.npy")

    ncells = int(np.genfromtxt(f"{modeldir}/ncells.txt"))
    sigma = float(np.genfromtxt(f"{modeldir}/sigma.txt"))

    best_idx = np.argmin(loss_hist_valid)
    model_fpath = f"{modeldir}/{modelname}_{best_idx}.pth"
    if verbosity: print(f"Best model: {model_fpath}")

    modelargs = load_args_from_log(f"{modeldir}/log_args.txt")
    if verbosity: print("Args:", modelargs)

    f_signal = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
    model = PhiNN(
        ndim=modelargs['ndims'], nsig=modelargs['nsigs'], 
        f_signal=f_signal, nsigparams=5,
        ncells=ncells,
        hidden_dims=modelargs['hidden_dims'],
        hidden_acts=modelargs['hidden_acts'],
        final_act=modelargs['final_act'],
        layer_normalize=modelargs['layer_normalize'],
        infer_noise=modelargs['infer_noise'],
        sigma=modelargs['sigma'],
        device='cpu',
        dtype=torch.float32 if modelargs['dtype']=='float32' else torch.float64,
        sample_cells=True,
    )

    if verbosity: print(f"Sigma: {np.exp(model.logsigma.item()):.4g}")

    model.load_state_dict(torch.load(model_fpath, 
                                     map_location=torch.device('cpu')))
    model.eval()

    return model, modelargs, loss_hist_train, loss_hist_valid

def plot_training_loss_history(loss_hist_train, saveas=None):
    fig, ax = plt.subplots(1, 1)
    ax.plot(1+np.arange(len(loss_hist_train)), loss_hist_train, '.-')
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"loss")
    ax.set_title(f"Training Loss")
    if saveas: plt.savefig(saveas)
    return ax

def plot_validation_loss_history(loss_hist_valid, saveas=None):
    fig, ax = plt.subplots(1, 1)
    ax.plot(1+np.arange(len(loss_hist_valid)), loss_hist_valid, '.-')
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"loss")
    ax.set_title(f"Validation Loss")
    if saveas: plt.savefig(saveas)
    return ax

def plot_loss_history(loss_hist_train, loss_hist_valid, saveas=None):
    fig, ax = plt.subplots(1, 1)
    ax.plot(1+np.arange(len(loss_hist_train)), loss_hist_train, 'r.-',
            label="Training")
    ax.plot(1+np.arange(len(loss_hist_valid)), loss_hist_valid, 'b.-',
            label="Validation")
    ax.set_xlabel(f"epoch")
    ax.set_ylabel(f"loss")
    ax.set_title(f"Loss History")
    if saveas: plt.savefig(saveas)
    return ax

def plot_train_vs_valid_history(loss_hist_train, loss_hist_valid, saveas=None,
                                log=True):
    fig, ax = plt.subplots(1, 1)
    if log:
        ax.loglog(loss_hist_train, loss_hist_valid, '.-')
    else:
        ax.plot(loss_hist_train, loss_hist_valid, '.-')
    ax.set_xlabel(f"Training loss")
    ax.set_ylabel(f"Validation loss")
    ax.set_title(f"Training vs Validation Loss")
    if saveas: plt.savefig(saveas)
    return ax

model, model_args, loss_hist_train, loss_hist_valid = load_model_directory(
    modeldir, model
)

if verbosity:
    print("*** Model Parameters ***")
    print(f"Sigma: {np.exp(model.logsigma.item()):.4g}")
    print(f"Tilt map:\n{list(model.tilt_nn.parameters())[0].detach().numpy()}")

plot_training_loss_history(
    loss_hist_train, 
    saveas=f"{outdir}/loss_hist_training.png"
)
plot_validation_loss_history(
    loss_hist_valid,
    saveas=f"{outdir}/loss_hist_validation.png"
)
plot_loss_history(
    loss_hist_train, 
    loss_hist_valid,
    saveas=f"{outdir}/loss_hist.png"
)
plot_train_vs_valid_history(
    loss_hist_train, 
    loss_hist_valid, 
    log=True,
    saveas=f"{outdir}/loss_train_vs_valid.png"
)

model.plot_phi(
    r=3, res=100, show=True, normalize=True, log_normalize=False,
    saveas=f"{outdir}/phi_heatmap.png"
)

signals_to_plot = [
    [0, 1], [1, 0], [1, 1],
]

for i, sig in enumerate(signals_to_plot):
    model.plot_f(
        signal=sig, r=3, res=20, show=True,
        title=f"$\\vec{{F}}(x,y|\\vec{{s}}=\\langle{sig[0]:.2g},{sig[1]:.2g}\\rangle)$",
        cbar_title="$|\\vec{F}|$",
        saveas=f"{outdir}/f_plot_sig_{i}.png"
    )
