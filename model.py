"""Neural Net Model

"""

import warnings
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd.functional import jacobian as jacobian

class PhiNN(nn.Module):

    _activation_keys = {
        'none' : None,
        'softplus' : nn.Softplus,
        'elu' : nn.ELU,
        'tanh' : nn.Tanh,
    }
    
    def __init__(self, 
            ndim=2, 
            nsig=2, 
            f_signal=None, 
            nsigparams=5,
            ncells=100,
            sample_cells=False,
            hidden_dims=[16, 32, 32, 16],
            hidden_acts=nn.ELU,
            final_act=nn.Softplus,
            layer_normalize=False,
            infer_noise=False,
            sigma=1e-3,
            include_signal_bias=False,
            init_weights=True,
            init_weight_values_phi=None,
            init_weight_values_tilt=None,
            device='cpu',
            dtype=torch.float32,
            rng=None,
            testing=False,
            testing_dw=None,
            ):
        """
        Args:
            ndim: (int) dimension of state space.
            nsig: (int) number of signals.
            f_signal: (callable) signal function mapping time and signal 
                parameters to the value of each signal.
        """
        super().__init__()
        self.ndim = ndim
        self.nsig = nsig
        self.f_signal = f_signal
        self.nsigparams = nsigparams
        self.ncells = ncells
        self.infer_noise = infer_noise
        self.sigma = sigma
        self.device = device
        self.dtype = dtype
        self.sample_cells = sample_cells
        self.rng = rng if rng else np.random.default_rng()
        self.testing = testing
        self.testing_dw = testing_dw

        # Apply testing methods
        if self.testing:
            self._sample_dw = self._sample_dw_test_version

        # Potential Network hidden layer specifications
        hidden_dims, hidden_acts, final_act = self._check_hidden_layers(
            hidden_dims=hidden_dims, 
            hidden_acts=hidden_acts, 
            final_act=final_act,
        )
        self.hidden_dims = hidden_dims
        self.hidden_acts = hidden_acts
        self.final_act = final_act
        
        # Noise inference or constant
        if self.infer_noise:
            self.logsigma = torch.nn.Parameter(torch.tensor(np.log(sigma)))
        else:
            if self.device != 'cpu':
                self.sigma = torch.tensor(sigma, dtype=self.dtype, 
                                          device=device)

        # Potential Neural Network: Maps ndims to a scalar.
        self.phi_nn = self._construct_phi_nn(
            hidden_dims, hidden_acts, final_act, 
            layer_normalize, testing=testing
        )
        
        # Tilt Neural Network: Linear tilt values. Maps nsigs to ndims.
        self.tilt_nn = self._construct_tilt_nn(
            include_signal_bias=include_signal_bias
        )

        # Initialize model parameters
        if init_weights:
            self.initialize_weights(
                testing=testing, 
                vals_phi=init_weight_values_phi,
                vals_tilt=init_weight_values_tilt,
                include_signal_bias=include_signal_bias,
            )
        
        # Summed version of phi for computing gradients efficiently
        self._phi_summed = lambda y : self.phi_nn(y).sum(axis=0)

    ######################
    ##  Getter Methods  ##
    ######################
    
    def get_ncells(self):
        if isinstance(self.ncells, torch.Tensor):
            return self.ncells.item()
        return self.ncells
    
    def get_sigma(self):
        if self.infer_noise:
            return np.exp(self.logsigma.data.cpu().numpy())
        if isinstance(self.sigma, torch.Tensor):
            return self.sigma.item()
        return self.sigma

    ##############################
    ##  Core Landscape Methods  ##
    ##############################

    def f(self, t, y, sig_params):
        """Drift term. 
        
        Applies across batches.

        Args:
            t (Tensor) : Time values. Shape (b,).
            y (Tensor) : State vector. Shape (b,n,d).
            sig_params (Tensor) : Signal parameters. Shape (b,nsigparams).
        Returns:
            Tensor of shape (b,n,d).
        """
        return -(self.grad_phi(t, y) + self.grad_tilt(t, sig_params)[:,None,:])

    def g(self, t, y):
        """Diffusion term. 
        
        Currently only implements scalar noise level. Applies across batches. 
        (TODO: generalize noise term.)

        Args:
            t (Tensor) : Time values. Shape (b,).
            y (Tensor) : State vector. Shape (b,n,d).
        Returns:
            Tensor of shape (b,n,d).
        """
        if self.infer_noise:
            return torch.exp(self.logsigma) * torch.ones(
                y.shape, dtype=self.dtype, device=self.device)
        else:
            return self.sigma * torch.ones(
                y.shape, dtype=self.dtype, device=self.device)
    
    def phi(self, y):
        """Potential value, without tilt. 
        
        Applies across batches.

        Args:
            y (Tensor) : State vector. Shape (b,n,d) or (n, d).
        Returns:
            Tensor of shape (b, n, 1) or (n, 1).
        """
        return self.phi_nn(y)

    def grad_phi(self, t, y):
        """Gradient of potential, without tilt. 
        
        Applies across batches.

        Args:
            t (Tensor) : Time values. Shape (b,).
            y (Tensor) : State vector. Shape (b,n,d).
        Returns:
            Tensor of shape (b,n,d).
        """
        return jacobian(self._phi_summed, y, create_graph=True).sum(axis=(0,1))
        
    def grad_tilt(self, t, sig_params):
        """Gradient of linear tilt function. 
        
        Applies across batches.

        Args:
            t (Tensor) : Time values. Shape (b,).
            sig_params (Tensor) : Signal parameters. Shape (b,nsigparams).
        Returns:
            Tensor of shape (b,nsigs).
        """
        signal_vals = self.f_signal(t, sig_params)
        return self.tilt_nn(signal_vals)
    
    #########################
    ##  Evolution Methods  ##
    #########################

    def step(self, t, y, sig_params, dt, dw):
        """Update states taking a step via the Euler-Maruyama method. 
        
        Applies across batches.

        Args:
            t (Tensor) : Time values. Shape (b,).
            y (Tensor) : State vector. Shape (b,n,d).
            sig_params (Tensor) : Signal parameters. Shape (b,nsigparams).
            dt (Tensor) : Timesteps. Shape (b,).
            dw (Tensor) : Wiener increments, i.e. random draws from 
                Normal(mu=0,var=dt). Shape (b,n,d).
        Returns:
            Tensor of shape (b,n,d).
        """
        fval = self.f(t, y, sig_params)
        gval = self.g(t, y)
        return y + fval * dt + gval * dw
    
    def simulate_forward(self, t0, t1, y0, sigparams, dt, history=False):
        """Simulate all trajectories forward in time.

        Applies across batches. Assumes that while the start and end times may
        differ across batches, their differences are uniform, i.e. t1 - t0 = C,
        a common temporal interval length. In addition, a uniform timestep is 
        used across batches.

        Args:
            t0 (Tensor) : Start times. Shape (b,).
            t1 (Tensor) : End times. Shape (b,).
            y0 (Tensor) : Initial states. Shape (b,n,d).
            sig_params (Tensor) : Signal parameters. Shape (b,nsigparams).
            dt (float) : Euler-Maruyama timestep. Used across all batches.
            history (bool) : Flag indicating whether to store and return state 
                history. Default False.
        Returns:
            Tensor of shape (b,n,d) or, if history=True, a tuple containing 
            this and a list of states of the same shape.
        """
        
        ts = torch.ones(t0.shape, device=self.device) * t0
        nsteps = round((t1[0].item() - t0[0].item()) / dt)
        y = y0
        if history:
            y_hist = []
            y_hist.append(y0.detach().numpy())
            for i in range(nsteps):
                dw = self._sample_dw(dt, y0.shape, self.device, self.dtype)
                y = self.step(ts, y, sigparams, dt, dw)
                ts += dt
                y_hist.append(y.detach().numpy())

            return y, y_hist
        else:
            for i in range(nsteps):
                dw = self._sample_dw(dt, y0.shape, self.device, self.dtype)
                y = self.step(ts, y, sigparams, dt, dw)
                ts += dt
            return y

    def forward(self, x, dt=1e-3):
        """Forward pass of the model.

        Applies across batches.

        Args:
            x (Tensor): Model inputs, in batches. Each batch consists of an
                initial and final time, a number n of d-dimensional cells, 
                flattened into a 1-d array, and a number of signal parameters.
                Shape (b,2+d*n+nsigparams).
            dt (float): Constant step size. Default 1e-3.

        Returns:
            Tensor of shape (b,n,d). The final state of each cell ensemble, 
            across batches.
        """
        # Parse the inputs
        t0 = x[...,0]
        t1 = x[...,1]
        y0 = x[...,2:-self.nsigparams].view([len(x), -1, self.ndim])
        sigparams = x[...,-self.nsigparams:]

        # Sample from the given set of cells, or use the sample directly.
        if self.sample_cells:
            y0 = self._sample_y0(y0)
        
        if y0.shape[1] != self.ncells:
            msg = f"Simulating {y0.shape[1]} cells. Expected {self.ncells}."
            raise RuntimeError(msg)
        
        y0.requires_grad_()  # Must be able to compute spatial gradients.
        return self.simulate_forward(t0, t1, y0, sigparams, dt)
    
    ##############################
    ##  Initialization Methods  ##
    ##############################

    def _check_hidden_layers(self, hidden_dims, hidden_acts, final_act):
        """Check the model architecture.
        """
        nlayers = len(hidden_dims)
        # Convert singular hidden activation to a list.
        if hidden_acts is None or isinstance(hidden_acts, (str, type)):
            hidden_acts = [hidden_acts] * nlayers
        elif isinstance(hidden_acts, list) and len(hidden_acts) == 1:
            hidden_acts = hidden_acts * nlayers
        # Check number of hidden activations
        if len(hidden_acts) != nlayers:
            msg = "Number of activation functions must match number of " + \
                  f"hidden layers. Got {hidden_acts} for {nlayers} layers."
            raise RuntimeError(msg)
        # Check hidden activation types
        for i, val in enumerate(hidden_acts):
            if isinstance(val, str):
                hidden_acts[i] = self._activation_keys[val.lower()]
            elif isinstance(val, type) or val is None:
                pass
            else:
                msg = f"Cannot handle activation specs: {hidden_acts}"
                raise RuntimeError(msg)
        # Check final activation types
        if isinstance(final_act, str):
            final_act = self._activation_keys[final_act.lower()]
        elif isinstance(final_act, type) or final_act is None:
                pass
        else:
            msg = f"Cannot handle final activation spec: {final_act}"
            raise RuntimeError(msg)
        return hidden_dims, hidden_acts, final_act
    
    def _add_layer(self, layer_list, din, dout, activation, normalization):
        layer_list.append(nn.Linear(din, dout, dtype=self.dtype))
        if normalization:
            layer_list.append(nn.LayerNorm([dout]))
        if activation:
            layer_list.append(activation())
        

    def _construct_phi_nn(self, hidden_dims, hidden_acts, final_act, 
                          layer_normalize, testing):
        if testing:
            return nn.Sequential(
                nn.Linear(self.ndim, 3, bias=False),
                nn.Tanh(),
                nn.Linear(3, 3, bias=False),
                nn.Tanh(),
                nn.Linear(3, 1, bias=False),
            )
        
        layer_list = []
        # Hidden layers
        self._add_layer(
            layer_list, self.ndim, hidden_dims[0], 
            activation=hidden_acts[0], 
            normalization=layer_normalize
        )
        for i in range(len(hidden_dims) - 1):
            self._add_layer(
                layer_list, hidden_dims[i], hidden_dims[i+1], 
                activation=hidden_acts[i+1], 
                normalization=layer_normalize
            )
        # Final layer
        self._add_layer(
            layer_list, hidden_dims[-1], 1,  
            activation=final_act, 
            normalization=False
        )
        return nn.Sequential(*layer_list)
    
    def _construct_tilt_nn(self, include_signal_bias):
        layer_list = [nn.Linear(self.nsig, self.ndim, bias=include_signal_bias, 
                                dtype=self.dtype)]
        return nn.Sequential(*layer_list)

    def initialize_weights(self, testing=False, vals_phi=None, vals_tilt=None,
                           include_signal_bias=False):
        if testing:
            self._initialize_test_weights(vals_phi, vals_tilt)
            return
        # Weight initialization for Phi
        for layer in self.phi_nn:
            if isinstance(layer, nn.Linear):
                # nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)
        # Weight initialization for Tilt
        for layer in self.tilt_nn:
            if isinstance(layer, nn.Linear):
                # nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.xavier_uniform_(layer.weight)
                if include_signal_bias:
                    nn.init.constant_(layer.bias, 0.01)
        # Weight initialization for Sigma
        # if self.infer_noise:
        #     nn.init.constant_(self.logsigma, self)
        
    def _initialize_test_weights(self, vals_phi=None, vals_tilt=None):
        # Initialize weights for Phi Net
        if vals_phi is not None:
            count = 0
            for layer in self.phi_nn:
                if isinstance(layer, nn.Linear):
                    w = torch.tensor(vals_phi[count], dtype=self.dtype,
                                     device=self.device)
                    layer.weight = torch.nn.Parameter(w)
                    count += 1
        # Initialize weights for Tilt Net
        if vals_tilt is not None:
            count = 0
            for layer in self.tilt_nn:
                if isinstance(layer, nn.Linear):
                    w = torch.tensor(vals_tilt[count], dtype=self.dtype,
                                     device=self.device)
                    layer.weight = torch.nn.Parameter(w)
                    count += 1
    
    ########################
    ##  Plotting Methods  ##
    ########################

    def plot_phi(self, r=4, res=50, plot3d=False, **kwargs):
        """Plot the scalar function phi.
        Args:
            r (int) : 
            res (int) :
            plot3d (bool) :
            normalize (bool) :
            log_normalize (bool) :
            ax (Axis) :
            figsize (tuple[float]) :
            xlims (tuple[float]) :
            ylims (tuple[float]) :
            xlabel (str) :
            ylabel (str) :
            zlabel (str) :
            title (str) :
            cmap (Colormap) :
            include_cbar (bool) :
            cbar_title (str) :
            cbar_titlefontsize (int) :
            cbar_ticklabelsize (int) :
            view_init (tuple) :
            saveas (str) :
            show (bool) :
        Returns:
            Axis object.
        """
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        normalize = kwargs.get('normalize', True)
        log_normalize = kwargs.get('log_normalize', True)
        ax = kwargs.get('ax', None)
        figsize = kwargs.get('figsize', (6, 4))
        xlims = kwargs.get('xlims', None)
        ylims = kwargs.get('ylims', None)
        xlabel = kwargs.get('xlabel', "$x$")
        ylabel = kwargs.get('ylabel', "$y$")
        zlabel = kwargs.get('zlabel', "$\\phi$")
        title = kwargs.get('title', "$\\phi(x,y)$")
        cmap = kwargs.get('cmap', 'coolwarm')
        include_cbar = kwargs.get('include_cbar', True)
        cbar_title = kwargs.get('cbar_title', 
                                "$\\ln\\phi$" if log_normalize else "$\\phi$")
        cbar_titlefontsize = kwargs.get('cbar_titlefontsize', 10)
        cbar_ticklabelsize = kwargs.get('cbar_ticklabelsize', 8)
        view_init = kwargs.get('view_init', (30, -45))
        saveas = kwargs.get('saveas', None)
        show = kwargs.get('show', False)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if self.training:
            warnings.warn("Not plotting. Currently training=True.")
            return
        if ax is None and plot3d:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        elif ax is None and (not plot3d):
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Compute phi
        x = np.linspace(-r, r, res)
        y = np.linspace(-r, r, res)
        xs, ys = np.meshgrid(x, y)
        z = np.array([xs.flatten(), ys.flatten()]).T[None,...]
        z = torch.tensor(z, requires_grad=True, 
                         dtype=torch.float32, device=self.device)
        phi = self.phi(z).detach().cpu().numpy()  # move to cpu
        
        # Normalization
        if normalize:
            phi = 1 + phi - phi.min()  # set minimum to 1
        if log_normalize:
            phi = np.log(phi)

        # Plot phi
        if plot3d:
            sc = ax.plot_surface(xs, ys, phi.reshape(xs.shape), cmap=cmap)
        else:
            sc = ax.pcolormesh(
                xs, ys, phi.reshape(xs.shape),
                cmap=cmap, 
                vmin=phi.min(), vmax=phi.max()
            )
        # Colorbar
        if include_cbar:
            cbar = plt.colorbar(sc)
            cbar.ax.set_title(cbar_title, size=cbar_titlefontsize)
            cbar.ax.tick_params(labelsize=cbar_ticklabelsize)
        
        # Format plot
        if xlims is not None: ax.set_xlim(*xlims)
        if ylims is not None: ax.set_ylim(*ylims)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if plot3d: 
            ax.set_zlabel(zlabel)
            ax.view_init(*view_init)
        plt.tight_layout()
        
        # Save and close
        if saveas: plt.savefig(saveas, bbox_inches='tight')
        if not show: plt.close()
        return ax
    
    def plot_f(self, signal=0, r=4, res=50, **kwargs):
        """Plot the vector field f.
        Args:
            signal (float or tuple[float]) :
            r (int) : 
            res (int) :
            ax (Axis) :
            figsize (tuple[float]) :
            xlims (tuple[float]) :
            ylims (tuple[float]) :
            xlabel (str) :
            ylabel (str) :
            title (str) :
            cmap (Colormap) :
            include_cbar (bool) :
            cbar_title (str) :
            cbar_titlefontsize (int) :
            cbar_ticklabelsize (int) :
            saveas (str) :
            show (bool) :
        Returns:
            Axis object.
        """
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        ax = kwargs.get('ax', None)
        figsize = kwargs.get('figsize', (6, 4))
        xlims = kwargs.get('xlims', None)
        ylims = kwargs.get('ylims', None)
        xlabel = kwargs.get('xlabel', "$x$")
        ylabel = kwargs.get('ylabel', "$y$")
        title = kwargs.get('title', "$f(x,y|\\vec{s})$")
        cmap = kwargs.get('cmap', 'coolwarm')
        include_cbar = kwargs.get('include_cbar', True)
        cbar_title = kwargs.get('cbar_title', "$|f|$")
        cbar_titlefontsize = kwargs.get('cbar_titlefontsize', 10)
        cbar_ticklabelsize = kwargs.get('cbar_ticklabelsize', 8)
        saveas = kwargs.get('saveas', None)
        show = kwargs.get('show', False)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if self.training:
            warnings.warn("Not plotting. Currently training=True.")
            return
        if ax is None: fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Initialize signal parameters TODO: don't hard-code the parameters
        signal_params = torch.tensor([[1, *signal, *signal]], 
                                     dtype=self.dtype, device=self.device)
        eval_time = torch.zeros(1, dtype=self.dtype, device=self.device)
        
        # Compute f
        x = np.linspace(-r, r, res)
        y = np.linspace(-r, r, res)
        xs, ys = np.meshgrid(x, y)
        z = np.array([xs.flatten(), ys.flatten()]).T[None,...]
        z = torch.tensor(z, requires_grad=True, 
                         dtype=self.dtype, device=self.device)
        f = self.f(eval_time, z, signal_params).detach().cpu().numpy()
        fu, fv = f.T
        fnorms = np.sqrt(fu**2 + fv**2)

        # Plot force field, tilted by signals
        sc = ax.quiver(xs, ys, fu/fnorms, fv/fnorms, fnorms, cmap=cmap)
        
        # Colorbar
        if include_cbar:
            cbar = plt.colorbar(sc)
            cbar.ax.set_title(cbar_title, size=cbar_titlefontsize)
            cbar.ax.tick_params(labelsize=cbar_ticklabelsize)
        
        # Format plot
        if xlims is not None: ax.set_xlim(*xlims)
        if ylims is not None: ax.set_ylim(*ylims)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        
        # Save and close
        if saveas: plt.savefig(saveas, bbox_inches='tight')
        if not show: plt.close()
        return ax

    ######################
    ##  Helper Methods  ##
    ######################

    def _sample_dw(self, dt, shape, device, dtype):
        return math.sqrt(dt) * torch.normal(0, 1, shape, 
                                            device=device, dtype=dtype)

    def _sample_dw_test_version(self, dt, shape, device, dtype):
        dw = math.sqrt(dt) * torch.normal(0, 1, shape, 
                                            device=device, dtype=dtype)
        if self.testing_dw is not None:
            dw[:] = self.testing_dw
        return dw

    def _sample_y0(self, y0):
        y0_samp = torch.empty([y0.shape[0], self.ncells, y0.shape[2]], 
                               dtype=self.dtype, device=self.device)
        if y0.shape[1] < self.ncells:
            # Sample with replacement
            for bidx in range(y0.shape[0]):
                samp_idxs = torch.tensor(
                    self.rng.choice(y0.shape[1], self.ncells, True),
                    dtype=int, device=self.device
                )
                y0_samp[bidx,:] = y0[bidx,samp_idxs]
        else:
            # Sample without replacement
            for bidx in range(y0.shape[0]):
                samp_idxs = torch.tensor(
                    self.rng.choice(y0.shape[1], self.ncells, False),
                    dtype=int, device=self.device
                )
                y0_samp[bidx,:] = y0[bidx,samp_idxs]
        return y0_samp
        