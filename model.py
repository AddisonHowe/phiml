"""Neural Net Model

"""

import numpy as np
import torch
from torch import nn
from torch.autograd.functional import jacobian as jacobian

class PhiNN(nn.Module):
    
    def __init__(self, ndim=2, nsig=2, f_signal=None, nsigparams=5, **kwargs):
        """
        Args:
            ndim: (int) dimension of state space.
            nsig: (int) number of signals.
            f_signal: (callable) signal function mapping time and signal 
                parameters to the value of each signal.
        """
        super().__init__()
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        ncells = kwargs.get('ncells', 100)
        infer_noise = kwargs.get('infer_noise', False)
        sigma = kwargs.get('sigma', 1e-3)
        testing = kwargs.get('testing', False)
        include_signal_bias = kwargs.get('include_bias', False)
        init_weights = kwargs.get('init_weights', True)
        init_weight_values_phi = kwargs.get('init_weight_values_phi', None)
        init_weight_values_tilt = kwargs.get('init_weight_values_tilt', None)
        testing_dw = kwargs.get('testing_dw', None)
        device = kwargs.get('device', 'cpu')
        sample_cells = kwargs.get('sample_cells', False)
        dtype = kwargs.get('dtype', torch.float32)
        rng = kwargs.get('rng', np.random.default_rng())
        hidden_dims = kwargs.get('hidden_dims', [16, 32, 32, 16])
        layer_normalize = kwargs.get('layer_normalize', False)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
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
        self.rng = rng
        self.testing = testing
        self.testing_dw = testing_dw

        if self.infer_noise:
            self.logsigma = torch.nn.Parameter(torch.tensor(np.log(sigma)))
        else:
            if self.device != 'cpu':
                self.sigma = torch.tensor(sigma, dtype=self.dtype, device=device)

        # Potential Neural Network: Maps ndims to a scalar. 
        activation1 = nn.Tanh if testing else nn.Softplus
        if testing:
            self.phi_nn = nn.Sequential(
                nn.Linear(ndim, 3, bias=False),
                activation1(),
                nn.Linear(3, 3, bias=False),
                activation1(),
                nn.Linear(3, 1, bias=False),
            )
        else:
            layer_list = [nn.Linear(ndim, hidden_dims[0])]
            if layer_normalize:
                layer_list.append(nn.LayerNorm([hidden_dims[0]]))
            layer_list.append(activation1())

            for i in range(len(hidden_dims) - 1):
                layer_list.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                if layer_normalize:
                    layer_list.append(nn.LayerNorm([hidden_dims[i+1]]))
                layer_list.append(activation1())
            
            layer_list.append(nn.Linear(hidden_dims[-1], 1))

            self.phi_nn = nn.Sequential(*layer_list)

        # Tilt Neural Network: Linear tilt values. Maps nsigs to ndims.
        self.tilt_nn = nn.Sequential(
            nn.Linear(self.nsig, self.ndim, bias=include_signal_bias)
        )

        if init_weights:
            self.initialize_weights(
                testing=testing, 
                vals_phi=init_weight_values_phi,
                vals_tilt=init_weight_values_tilt,
                include_signal_bias=include_signal_bias,
            )
            
        self._phi_summed = lambda y : self.phi_nn(y).sum(axis=0)

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

    def f(self, t, y, sig_params):
        """Drift term.
        Args:
            t : Scalar time.
            y : State vector of shape (k, ndim,).
        Returns:
            tensor of shape (k, ndim,)
        """
        # t1 = self.grad_phi(t, y)
        # t2 = self.grad_tilt(t, sig_params)[:,None,:]
        # assert not torch.any(torch.isnan(t1)), t1
        # assert not torch.any(torch.isnan(t2)), t2
        return -(self.grad_phi(t, y) + self.grad_tilt(t, sig_params)[:,None,:])

    def g(self, t, y):
        """Diffusion term.
        Args:
            t : Scalar time.
            y : State vector of shape (k, ndim,).
        Returns:
            Scalar noise or TODO: nonscalar noise
        """
        if self.infer_noise:
            return torch.exp(self.logsigma)
        else:
            return self.sigma
    
    def phi(self, y):
        """Potential value, without tilt.
        Args:
            y : State vector of shape (k, ndim,).
        Returns:
            scalar
        """
        return self.phi_nn(y)

    def grad_phi(self, t, y):
        """Gradient of potential, without tilt.
        Args:
            t : Scalar time.
            y : State vector of shape (ndim,).
        Returns:
            tensor of shape (ndim,)
        """
        grad = jacobian(self._phi_summed, y, create_graph=True).sum(axis=(0,1))
        return grad
        
    def grad_tilt(self, t, sig_params):
        """Gradient of tilt function.
        Args:
            t : Scalar time.
            sig_params : tensor of shape (nsig_params,) : Signal parameters.
        Returns:
            tensor of shape (nsigs,)
        """
        signal_vals = self.f_signal(t, sig_params)
        return self.tilt_nn(signal_vals)

    def step(self, t, y, params, dt, dw):
        """Take an Euler-Maruyama step.
        Args:
            t : Current time (scalar).
            y : State vector of shape (ndim,).
            params : parameters of the signal function.
            dt : timestep.
            dw : Tensor of shape (ndim,). Wiener increment, i.e. random draws 
                from Normal(mu=0, var=dt).   
        Returns:
            Updated state. Tensor of shape (ndim,)
        """
        fval = self.f(t, y, params)
        gval = self.g(t, y)
        # assert not (torch.any(torch.isnan(fval)) or torch.any(torch.isnan(gval))), f"{gval}\n{fval}\n{y}"
        return y + fval * dt + gval * dw
    
    def forward(self, x, dt=1e-3):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Model inputs. Shape (batch_size, 2+ncells+nparams)
            dt (float, optional): Constant step size. Defaults to 1e-3.

        Returns:
            _type_: _description_
        """
        # Parse the inputs
        t0 = x[...,0]
        t1 = x[...,1]
        y0 = x[...,2:-self.nsigparams].view([len(x), -1, self.ndim])
        sigparams = x[...,-self.nsigparams:]
        # Sample from the given set of cells, or use the sample directly.
        if self.sample_cells:
            y0 = self._sample_y0(y0)
        
        assert y0.shape[1] == self.ncells, \
            f"Trying to simulate {y0.shape[1]} cells. Expected {self.ncells}."
        
        y0.requires_grad_()
        return self.simulate_forward(t0, t1, y0, sigparams, dt=dt)

    def simulate_forward(self, t0, t1, y0, sigparams, 
                         dt=1e-3, history=False):
        """Simulate all trajectories forward in time.
        Args:
            t0 (batch_size): 
            t1 (batch_size): 
            y0 (batch_size, ncells, ndim): 
            sigparams (): 
        Returns:
            ...
        """
                
        ts = torch.ones(t0.shape, device=self.device) * t0
        nsteps = round((t1[0].item() - t0[0].item()) / dt)
        
        y = y0
        y_hist = []

        if history:
            y_hist.append(y0.detach().numpy())
        for i in range(nsteps):
            dw = np.sqrt(dt) * torch.normal(0, 1, y0.shape, 
                                            device=self.device, 
                                            dtype=self.dtype)
            if self.testing and (self.testing_dw is not None):
                dw[:] = self.testing_dw
            y = self.step(ts, y, sigparams, dt, dw)
            ts += dt
            if history:
                y_hist.append(y.detach().numpy())
        return y if not history else (y, y_hist)
    
    def initialize_weights(self, testing=False, vals_phi=None, vals_tilt=None,
                           include_signal_bias=False):
        if testing:
            self._initialize_test_weights(vals_phi, vals_tilt)
            return
        # Weight initialization for Phi
        for layer in self.phi_nn:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)
        # Weight initialization for Tilt
        for layer in self.tilt_nn:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                if include_signal_bias:
                    nn.init.constant_(layer.bias, 0)
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
        
