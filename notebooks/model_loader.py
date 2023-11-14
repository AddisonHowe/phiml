import numpy as np
import torch
from phiml.model import PhiNN
from phiml.helpers import jump_function

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

def load_model_directory(modeldir, modelname, verbosity=1):
    loss_hist_train = np.load(f"{modeldir}/training_loss_history.npy")
    loss_hist_valid = np.load(f"{modeldir}/validation_loss_history.npy")

    ncells = int(np.genfromtxt(f"{modeldir}/ncells.txt"))
    sigma = float(np.genfromtxt(f"{modeldir}/sigma.txt"))

    best_idx = np.argmin(loss_hist_valid)
    model_fpath = f"{modeldir}/{modelname}_{best_idx}.pth"
    if verbosity > 0: print(f"Best model: {model_fpath}")

    modelargs = load_args_from_log(f"{modeldir}/log_args.txt")

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
    
    model.load_state_dict(torch.load(model_fpath, 
                                     map_location=torch.device('cpu')))
    model.eval();

    return model, modelargs, loss_hist_train, loss_hist_valid
