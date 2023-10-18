import pytest
import numpy as np
import torch
from model import PhiNN
from helpers import get_binary_function
from helpers import jump_function, mean_cov_loss

#####################
##  Configuration  ##
#####################

# Configuration settings...

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

W1 = np.array([
    [1, 3],
    [2, 2],
    [3, 1],
], dtype=float)

W2 = np.array([
    [1, 1, -2],
    [0, 1, 0],
    [-1, 2, 1],
], dtype=float)

W3 = np.array([
    [2, 3, 1]
], dtype=float)

WT1 = np.array([
    [2, 4],
    [-1, 1],
], dtype=float)

@pytest.mark.parametrize("ws, x, phi_exp", [
    [[W1, W2, W3], [[0, 0]], [0.0]],
    [[W1, W2, W3], [[0, 1]], [3.9934]],
    [[W1, W2, W3], [[1, 0]], [2.69506]],
    [[W1, W2, W3], [[1, 1]], [3.24787]],
])
def test_phi(ws, x, phi_exp):
    model = PhiNN(
        ndim=2, nsig=2, f_signal=None, 
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi = ws
    )

    x = torch.tensor(x, dtype=torch.float32)
    phi_act = model.phi(x).detach().numpy()
    assert np.allclose(phi_exp, phi_act, atol=1e-6)

@pytest.mark.parametrize("ws, x, grad_phi_exp", [
    [[W1, W2, W3], [[0, 0]], [[6.0, 14.0]]],
    [[W1, W2, W3], [[0, 1]], [[-3.5586, -0.83997]]],
    [[W1, W2, W3], [[1, 0]], [[1.11945, 2.71635]]],
    [[W1, W2, W3], [[1, 1]], [[-0.00409335, 0.0116181]]],
])
def test_grad_phi(ws, x, grad_phi_exp):
    model = PhiNN(
        ndim=2, nsig=2, f_signal=None, 
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi = ws
    )

    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    grad_phi_act = model.grad_phi(0, x).detach().numpy()
    assert np.allclose(grad_phi_exp, grad_phi_act, atol=1e-6)

@pytest.mark.parametrize('tcrit, p0, p1, t, signal_exp', [
    [5, [0, 1], [1, -1], 0, [0, 1]],
    [5, [0, 1], [1, -1], 1, [0, 1]],
    [5, [0, 1], [1, -1], 4, [0, 1]],
    [5, [0, 1], [1, -1], 5, [1, -1]],
    [5, [0, 1], [1, -1], 6, [1, -1]],
    [5, [0, 1], [1, -1], 10, [1, -1]],
    [5, [0, 1], [1, -1], 12, [1, -1]],
])
def test_binary_signal_function(tcrit, p0, p1, t, signal_exp):
    sigfunc = get_binary_function(tcrit, p0, p1)
    signal_act = sigfunc(t)
    assert np.allclose(signal_act, signal_exp)

@pytest.mark.parametrize('wts, tcrit, p0, p1, t, grad_tilt_exp', [
    [[WT1], 5, [0, 1], [1, -1], 0, [4, 1]],
    [[WT1], 5, [0, 1], [1, -1], 10, [-2, -2]],
])
def test_grad_tilt(wts, tcrit, p0, p1, t, grad_tilt_exp):
    sigparams = torch.tensor([tcrit, *p0, *p1], dtype=torch.float32)
    sigfunc = lambda t, p: jump_function(t, p[0], p[1:3], p[3:])
    model = PhiNN(
        ndim=2, nsig=2, f_signal=sigfunc, 
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_tilt = wts
    )
    grad_tilt_act = model.grad_tilt(t, sigparams).detach().numpy()
    assert np.allclose(grad_tilt_act, grad_tilt_exp)

@pytest.mark.parametrize('ws, wts, tcrit, p0, p1, t, x, f_exp', [
    [[W1, W2, W3], [WT1], 5, [0, 1], [1, -1], 0,  [[0, 1]], [[-0.441403, -0.16003]]],
    [[W1, W2, W3], [WT1], 5, [0, 1], [1, -1], 10, [[0, 1]], [[5.5586, 2.83997]]],
])
def test_f(ws, wts, tcrit, p0, p1, t, x, f_exp):
    sigparams = torch.tensor([tcrit, *p0, *p1], dtype=torch.float32)
    sigfunc = lambda t, p: jump_function(t, p[0], p[1:3], p[3:])
    model = PhiNN(
        ndim=2, nsig=2, f_signal=sigfunc, 
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi = ws,
        init_weight_values_tilt = wts,
    )
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    f_act = model.f(t, x, sigparams).detach().numpy()
    assert np.allclose(f_act, f_exp)
    
@pytest.mark.parametrize('ws, wts, tcrit, p0, p1, t, dt, sigma, dw, x, x_exp', [
    [[W1, W2, W3], [WT1], 5, [0, 1], [1, -1], 
     0,  1e-2, 1e-3, [[0.05, -0.024]], [[0, 1]], [[-0.00436403, 0.9983757]]
    ],
    [[W1, W2, W3], [WT1], 5, [0, 1], [1, -1], 
     10, 1e-2, 1e-3, [[0.05, -0.024]], [[0, 1]], [[0.055636, 1.0283757]]
    ],
    [[W1, W2, W3], [WT1], 5, [0, 1], [1, -1], 
     0, 1e-2, 1e-3, [[0.05, -0.024],[0.05, -0.024]], 
     [[0, 1], [0, 1]], [[-0.00436403, 0.9983757], [-0.00436403, 0.9983757]]
    ],
])
def test_step(ws, wts, tcrit, p0, p1, t, dt, sigma, dw, x, x_exp):
    sigparams = torch.tensor([tcrit, *p0, *p1], dtype=torch.float32)
    sigfunc = lambda t, p: jump_function(t, p[0], p[1:3], p[3:])
    model = PhiNN(
        ndim=2, nsig=2, f_signal=sigfunc, sigma=sigma,
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi = ws,
        init_weight_values_tilt = wts,
    )
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    dw = torch.tensor(dw, dtype=torch.float32)
    x_act = model.step(t, x, sigparams, dt, dw).detach().numpy()
    assert np.allclose(x_act, x_exp)
    
@pytest.mark.parametrize('ws, wts, tcrit, p0, p1, t, dt, tfin, \
                         sigma, dw, y', [
    [[W1, W2, W3], [WT1], 5, [0, 1], [1, -1], 0,  1e-2, 10,
     1e-3, [[0.05, -0.024]], [[0, 1]]
    ],
    [[W1, W2, W3], [WT1], 5, [0, 1], [1, -1], 10, 1e-2, 10,
     1e-3, [[0.05, -0.024]], [[0, 1]]
    ],
    [[W1, W2, W3], [WT1], 5, [0, 1], [1, -1], 0, 1e-2,  10,
     1e-3, [[0.05, -0.024],[0.05, -0.024]], [[0, 1], [0, 1]],
    ],
])
def test_simulate_forward(ws, wts, tcrit, p0, p1, t, dt, tfin, sigma, dw, y):
    ncells = len(y)
    sigfunc = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
    model = PhiNN(
        ndim=2, nsig=2, f_signal=sigfunc, sigma=sigma, ncells=ncells,
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi = ws,
        init_weight_values_tilt = wts,
        testing_dw=torch.tensor(dw, dtype=torch.float32),
    )
    y = np.array(y)
    x = np.concatenate([[t], [tfin], y.flatten(), [tcrit, *p0, *p1]])
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y_act = model.simulate_forward(x, dt=dt).detach().numpy()
    assert y_act.shape == (ncells, 2)
    

@pytest.mark.parametrize('y_sim, y_obs, loss_exp', [
    [   
        # Simulated
        [[[0,0],[1,1],[1,0],[0,1]],
         [[0,0],[0,0],[0,0],[0,0]],
         [[1,1],[1,1],[1,1],[1,1]]],
        # Observed
        [[[1,1],[0,0],[1,0],[0,1]],
         [[0,0],[0,0],[0,0],[0,0]],
         [[1,1],[1,1],[1,1],[1,1]]],
        # Loss
        (0 + 0 + 0)/3
    ],
    [   
        # Simulated
        [[[0,0],[1,1],[1,0],[0,1]],
         [[1,1],[1,1],[1,1],[1,1]],
         [[0,0],[0,0],[0,0],[0,0]]],
        # Observed
        [[[1,1],[0,0],[1,0],[0,1]],
         [[0,0],[0,0],[0,0],[0,0]],
         [[1,1],[1,1],[1,1],[1,1]]],
        # Loss
        ((0+0) + (2+0) + (2+0))/3
    ],
    [   
        # Simulated
        [[[0,0],[2,2],[2,0],[0,2]],  # mean: [1,1]  cov: [1.333,0,0,1.333]
         [[1,1],[1,1],[1,1],[1,1]],
         [[0,0],[0,0],[0,0],[0,0]]],
        # Observed
        [[[1,1],[0,0],[1,0],[0,1]],  # mean: [0.5,0.5]  cov: [0.333,0,0,0.333]
         [[0,0],[0,0],[0,0],[0,0]],
         [[1,1],[1,1],[1,1],[1,1]]],
        # Loss
        ((0.5+2) + (2+0) + (2+0))/3
    ],
])
def test_mean_cov_loss(y_sim, y_obs, loss_exp):
    y_sim = torch.Tensor(y_sim)
    y_obs = torch.Tensor(y_obs)
    loss_act = mean_cov_loss(y_sim, y_obs)
    assert np.allclose(loss_act, loss_exp)
