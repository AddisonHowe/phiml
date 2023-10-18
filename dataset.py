"""Dataset object for landscape simulations
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LandscapeSimulationDataset(Dataset):
    """A collection of data generated via landscape simulations.
    
    Each simulation consists of generating a number of paths between time
    t0=0 and t1, sampling the state of each path at a set of time intervals t_i,
    and thus capturing a pair (t_i, X_i), where X_i is an N by d state matrix.
    The data then consists of tuples (t_{i}, X_{i}, t_{i+1}, X_{i+1}), which 
    represent the evolution in state between consecutive sampling times.

    """
    
    def __init__(self, datdir, nsims, dim, 
                 transform=None, target_transform=None, **kwargs):
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        simprefix = kwargs.get('simprefix', 'sim')
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.nsims = nsims
        self.dim = dim
        self.transform = transform
        self.target_transform = target_transform
        self._load_data(datdir, nsims, simprefix=simprefix)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        t0, x0, t1, x1, ps = data
        # Transform input x
        if self.transform == 'tensor':
            x = np.concatenate([[t0], [t1], x0.flatten(), ps])
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        elif self.transform:
            x = self.transform(*data)
        else:
            x = t0, x0, t1, ps
        # Transform target y, the final distribution
        if self.target_transform == 'tensor':
            y = torch.tensor(x1, dtype=torch.float32)
        elif self.target_transform:
            y = self.target_transform(*data)
        else:
            y = x1
        return x, y

    ######################
    ##  Helper Methods  ##
    ######################
    
    def _load_data(self, datdir, nsims, simprefix='sim'):
        ts_all = []
        xs_all = []
        ps_all = []
        dataset = []
        for i in range(nsims):
            dirpath = f"{datdir}/{simprefix}{i}"
            assert os.path.isdir(dirpath), f"Not a directory: {dirpath}"
            ts = np.load(f"{dirpath}/ts.npy")
            xs = np.load(f"{dirpath}/xs.npy")
            ps = np.load(f"{dirpath}/ps.npy")
            p_params = np.load(f"{dirpath}/p_params.npy")
            ts_all.append(ts)
            xs_all.append(xs)
            ps_all.append(ps)
            self._add_sim_data_to_dataset(dataset, ts=ts, xs=xs, ps=ps, 
                                          p_params=p_params)
        self.dataset = dataset
        self.ts_all = ts_all
        self.xs_all = xs_all
        self.ps_all = ps_all

    def _add_sim_data_to_dataset(self, dataset, ts, xs, ps, p_params):
        ntimes = len(ts)
        for i in range(ntimes - 1):
            x0, x1 = xs[i], xs[i + 1]
            t0, t1 = ts[i], ts[i + 1]
            dataset.append((t0, x0, t1, x1, p_params))
