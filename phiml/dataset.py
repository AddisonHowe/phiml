"""Dataset object for landscape simulations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
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
        dtype = kwargs.get('dtype', torch.float32)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.nsims = nsims
        self.dim = dim
        self.transform = transform
        self.target_transform = target_transform
        self.dtype = dtype
        self._load_data(datdir, nsims, simprefix=simprefix)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        t0, x0, t1, x1, ps = data
        # Transform input x
        if self.transform == 'tensor':
            x = np.concatenate([[t0], [t1], x0.flatten(), ps])
            # x = torch.tensor(x, dtype=self.dtype, requires_grad=True)
            x = torch.tensor(x, dtype=self.dtype)
        elif self.transform:
            x = self.transform(*data)
        else:
            x = t0, x0, t1, ps
        # Transform target y, the final distribution
        if self.target_transform == 'tensor':
            y = torch.tensor(x1, dtype=self.dtype)
        elif self.target_transform:
            y = self.target_transform(*data)
        else:
            y = x1
        return x, y
    
    def preview(self, idx, **kwargs):
        """Plot a data item."""
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        ax = kwargs.get('ax', None)
        col1 = kwargs.get('col1', 'b')
        col2 = kwargs.get('col2', 'r')
        size = kwargs.get('size', 2)
        xlims = kwargs.get('xlims', None)
        ylims = kwargs.get('ylims', None)
        show = kwargs.get('show', True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        data = self.dataset[idx]
        t0, x0, t1, x1, ps = data
        if ax is None: fig, ax = plt.subplots(1, 1)
        ax.plot(x0[:,0], x0[:,1], '.', 
                c=col1, markersize=size, label=f"$t={t0:.3g}$")
        ax.plot(x1[:,0], x1[:,1], '.', 
                c=col2, markersize=size, label=f"$t={t1:.3g}$")
        if xlims is not None: ax.set_xlim(*xlims)
        if ylims is not None: ax.set_ylim(*ylims)
        ax.set_xlabel(f"$x$")
        ax.set_ylabel(f"$y$")
        ax.set_title(f"datapoint {idx}/{len(self)}")
        s = f"$t:{t0:.4g}\\to{t1:.4g}$\
            \n$t^*={ps[0]:.3g}$\
            \n$p_0=[{ps[1]:.3g}, {ps[2]:.3g}]$\
            \n$p_1=[{ps[3]:.3g}, {ps[4]:.3g}]$"
        ax.text(0.02, 0.02, s, fontsize=8, transform=ax.transAxes)
        ax.legend()
        if show: plt.show()
        return ax

    def animate(self, simidx, interval=50, **kwargs):
        """Animate a given simulation"""
        idx0 = int(simidx * len(self) // self.nsims)
        idx1 = idx0 + int(len(self) // self.nsims)
        video = []
        for idx in range(idx0, idx1):
            ax = self.preview(idx, **kwargs)
            ax.figure.canvas.draw()
            data = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
            video.append(data)
            plt.close()
        video = np.array(video)

        fig = plt.figure()
        plt.axis('off')
        plt.tight_layout()
        im = plt.imshow(video[0,:,:,:])
        plt.close() 
        def init():
            im.set_data(video[0,:,:,:])

        def ani_func(i):
            im.set_data(video[i,:,:,:])
            return im

        anim = animation.FuncAnimation(
            fig, ani_func, init_func=init, 
            frames=video.shape[0],
            interval=interval,
        )
        return anim.to_html5_video()

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
        self.dataset = np.array(dataset, dtype=object)
        self.ts_all = ts_all
        self.xs_all = xs_all
        self.ps_all = ps_all

    def _add_sim_data_to_dataset(self, dataset, ts, xs, ps, p_params):
        ntimes = len(ts)
        for i in range(ntimes - 1):
            x0, x1 = xs[i], xs[i + 1]
            t0, t1 = ts[i], ts[i + 1]
            dataset.append((t0, x0, t1, x1, p_params))
