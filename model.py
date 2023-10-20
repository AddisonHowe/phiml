"""Neural Net Model

"""

import numpy as np
import torch
from torch import nn

class PhiNN(nn.Module):
    
    def __init__(self, ndim=2, nsig=2, f_signal=None, **kwargs):
        """
        Args:
            ndim: (int) dimension of state space.
            nsig: (int) number of signals.
            f_signal: (callable) signal function mapping time and signal 
                parameters to the value of each signal.
        """
        super().__init__()
        #~~~~~~~~~~~~  process kwargs  ~~~~~~~~~~~~#
        sigma = kwargs.get('sigma', 1e-3)
        testing = kwargs.get('testing', False)
        include_bias = kwargs.get('include_bias', True)
        init_weights = kwargs.get('init_weights', True)
        init_weight_values_phi = kwargs.get('init_weight_values_phi', None)
        init_weight_values_tilt = kwargs.get('init_weight_values_tilt', None)
        self.testing_dw = kwargs.get('testing_dw', None)
        ncells = kwargs.get('ncells', 100)
        device = kwargs.get('device', 'cpu')
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        self.ndim = ndim
        self.nsig = nsig
        self.f_signal = f_signal
        self.sigma = sigma
        self.ncells = ncells
        self.testing = testing
        self.device = device

        if self.device != 'cpu':
            self.sigma = torch.tensor(sigma, dtype=torch.float32, device=device)

        # Potential Neural Network: Maps ndims to a scalar. 
        activation1 = nn.Tanh
        if testing:
            self.phi_nn = nn.Sequential(
                nn.Linear(ndim, 3, bias=False),
                activation1(),
                nn.Linear(3, 3, bias=False),
                activation1(),
                nn.Linear(3, 1, bias=False),
            )
        else:
            self.phi_nn = nn.Sequential(
                nn.Linear(ndim, 16),
                activation1(),
                nn.Linear(16, 32),
                activation1(),
                nn.Linear(32, 32),
                activation1(),
                nn.Linear(32, 16),
                activation1(),
                nn.Linear(16, 1),
            )

        # Tilt Neural Network: Linear tilt values. Maps nsigs to ndims.
        self.tilt_nn = nn.Sequential(
            nn.Linear(self.nsig, self.ndim, bias=include_bias)
        )

        if init_weights:
            self.initialize_weights(testing=testing, 
                                    vals_phi=init_weight_values_phi,
                                    vals_tilt=init_weight_values_tilt)

    def f(self, t, y, sig_params):
        """Drift term.
        Args:
            t : Scalar time.
            y : State vector of shape (k, ndim,).
        Returns:
            tensor of shape (k, ndim,)
        """
        return -(self.grad_phi(t, y) + self.grad_tilt(t, sig_params))

    def g(self, t, y):
        """Diffusion term.
        Args:
            t : Scalar time.
            y : State vector of shape (k, ndim,).
        Returns:
            Scalar noise or TODO: nonscalar noise
        """
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
        val_phi = self.phi_nn(y)
        grad = torch.autograd.grad(torch.sum(val_phi), y, 
                                   retain_graph=True, create_graph=True)[0]
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
        return y + fval * dt + gval * dw
    
    def forward(self, x, dt=1e-3):
        results = torch.zeros([len(x), self.ncells, self.ndim], dtype=torch.float32, device=self.device)
        for i in range(len(x)):
            y = self.simulate_forward(x[i], dt=dt)
            results[i] = y        
        return results

    def simulate_forward(self, x, dt=1e-3, history=False):
        """Simulate all trajectories forward in time.
        Args:
            x : tensor of shape ???
        Returns:
            ...
        """
        t0 = x[0]
        t1 = x[1]
        y0 = x[2:2+self.ndim*self.ncells].view([self.ncells, self.ndim])
        ps = x[2+self.ndim*self.ncells:]
        tcrit = ps[0]
        p0 = ps[1:3]
        p1 = ps[3:5]
        
        ts = torch.linspace(t0.item(), t1.item(), 1 + int((t1 - t0) / dt))
        y = y0
        y_hist = []
        sigparams = torch.tensor([tcrit, *p0, *p1], dtype=torch.float32, 
                                 device=self.device)
        if history:
            y_hist.append(y0.detach().numpy())
        for i, t in enumerate(ts):
            dw = torch.normal(0, np.sqrt(dt), [self.ncells, self.ndim], device=self.device)
            if self.testing and (self.testing_dw is not None):
                dw[:] = self.testing_dw
            y = self.step(t, y, sigparams, dt, dw)
            if history:
                y_hist.append(y.detach().numpy())
        return y if not history else (y, y_hist)
    
    def initialize_weights(self, testing=False, vals_phi=None, vals_tilt=None):
        if testing:
            self._initialize_test_weights(vals_phi, vals_tilt)
            return
        # Weight initialization scheme
        for layer in self.phi_nn:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)
        for layer in self.tilt_nn:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.1)
                nn.init.constant_(layer.bias, 0)
        
    def _initialize_test_weights(self, vals_phi=None, vals_tilt=None):
        # Initialize weights for Phi Net
        if vals_phi is not None:
            count = 0
            for layer in self.phi_nn:
                if isinstance(layer, nn.Linear):
                    w = torch.tensor(vals_phi[count], dtype=torch.float32,
                                     device=self.device)
                    layer.weight = torch.nn.Parameter(w)
                    count += 1
        # Initialize weights for Tilt Net
        if vals_tilt is not None:
            count = 0
            for layer in self.tilt_nn:
                if isinstance(layer, nn.Linear):
                    w = torch.tensor(vals_tilt[count], dtype=torch.float32,
                                     device=self.device)
                    layer.weight = torch.nn.Parameter(w)
                    count += 1
        
