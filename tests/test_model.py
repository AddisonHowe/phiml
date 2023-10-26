import pytest
import numpy as np
import torch
from model import PhiNN
from helpers import get_binary_function, jump_function
from helpers import mean_cov_loss, mean_diff_loss

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
        init_weight_values_phi = ws,
        dtype=torch.float64,
    )

    x = torch.tensor(x, dtype=torch.float64)
    phi_act = model.phi(x).detach().numpy()
    assert np.allclose(phi_exp, phi_act, atol=1e-6)

@pytest.mark.parametrize("ws, x, grad_phi_exp, shape_exp", [
    [[W1, W2, W3], [[[0, 0]]], [[[6.0, 14.0]]], (1, 1, 2)],
    [[W1, W2, W3], [[[0, 1]]], [[[-3.5586, -0.83997]]], (1, 1, 2)],
    [[W1, W2, W3], [[[1, 0]]], [[[1.11945, 2.71635]]], (1, 1, 2)],
    [[W1, W2, W3], [[[1, 1]]], [[[-0.00409335, 0.0116181]]], (1, 1, 2)],
    [[W1, W2, W3], 
     [[[1, 1]], [[1, 0]]], 
     [[[-0.00409335, 0.0116181]], [[1.11945, 2.71635]]], (2, 1, 2)],
    [[W1, W2, W3], 
     [[[1, 1], [0, 0]],  # 2 batches, 2 cells/batch
      [[1, 0], [0, 1]]], 
     [[[-0.00409335, 0.0116181], [6.0, 14.0]], 
      [[1.11945, 2.71635], [-3.5586, -0.83997]]], (2, 2, 2)],
    [[W1, W2, W3], 
     [[[1, 1], [0, 0]],  # 3 batches, 2 cells/batch
      [[1, 0], [0, 1]],
      [[1, 0], [0, 1]],], 
     [[[-0.00409335, 0.0116181], [6.0, 14.0]], 
      [[1.11945, 2.71635], [-3.5586, -0.83997]],
      [[1.11945, 2.71635], [-3.5586, -0.83997]]], (3, 2, 2)],
])
def test_grad_phi(ws, x, grad_phi_exp, shape_exp):
    model = PhiNN(
        ndim=2, nsig=2, f_signal=None, 
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi = ws,
        dtype=torch.float64,
    )

    x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    grad_phi_act = model.grad_phi(0, x).detach().numpy()

    errors = []
    if not np.allclose(grad_phi_exp, grad_phi_act, atol=1e-6):
        msg = f"Value mismatch between grad phi actual and expected."
        errors.append(msg)
    if not (grad_phi_act.shape == shape_exp):
        msg = f"Shape mismatch between grad phi actual and expected."
        msg += f"Expected {shape_exp}. Got {grad_phi_act.shape}."
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    

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

@pytest.mark.parametrize('wts, sigparams, t, grad_tilt_exp, shape_exp', [
    [[WT1], [[5, 0, 1, 1, -1]], [0], [[4, 1]], (1, 2)],
    [[WT1], [[5, 0, 1, 1, -1]], [10], [[-2, -2]], (1, 2)],
    [[WT1], 
     [[5, 0, 1, 1, -1],  # 3 batches
      [5, 0, 1, 1, -1],
      [5, 0, 1, 1, -1]], 
     [0, 10, 10], 
     [[4, 1], [-2, -2], [-2, -2]], 
     (3, 2)],
])
def test_grad_tilt(wts, sigparams, t, grad_tilt_exp, shape_exp):
    sigparams = torch.tensor(sigparams, dtype=torch.float64)
    sigfunc = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
    model = PhiNN(
        ndim=2, nsig=2, f_signal=sigfunc, 
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_tilt = wts,
        dtype=torch.float64,
    )
    ts = torch.tensor(t, dtype=torch.float64)
    grad_tilt_act = model.grad_tilt(ts, sigparams).detach().numpy()

    errors = []
    if not np.allclose(grad_tilt_act, grad_tilt_exp):
        msg = f"Value mismatch between grad tilt actual and expected."
        errors.append(msg)
    if not (grad_tilt_act.shape == shape_exp):
        msg = f"Shape mismatch between grad tilt actual and expected."
        msg += f"Expected {shape_exp}. Got {grad_tilt_act.shape}."
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize('ws, wts, sigparams, t, x, f_exp', [
    [[W1, W2, W3], [WT1], 
     [[5, 0, 1, 1, -1]], [0],  # 1 batch
     [[[0, 1]]], 
     [[[-0.441403, -0.16003]]]
    ],
    [[W1, W2, W3], [WT1], 
     [[5, 0, 1, 1, -1]], [10],  # 1 batch
     [[[0, 1]]], 
     [[[5.5586, 2.83997]]]
    ],
    [[W1, W2, W3], [WT1], 
     [[5, 0, 1, 1, -1],[5, 0, 1, 1, -1]], [0,10],  # 2 batches
     [[[0, 1]], [[0, 1]]], 
     [[[-0.441403, -0.16003]], [[5.5586, 2.83997]]]
    ],
])
def test_f(ws, wts, sigparams, t, x, f_exp):
    sigfunc = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
    model = PhiNN(
        ndim=2, nsig=2, f_signal=sigfunc, 
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi = ws,
        init_weight_values_tilt = wts,
        dtype=torch.float64,
    )
    sigparams = torch.tensor(sigparams, dtype=torch.float64)
    t = torch.tensor(t, dtype=torch.float64)
    x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
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
    sigparams = torch.tensor([[tcrit, *p0, *p1]], dtype=torch.float64)
    sigfunc = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
    model = PhiNN(
        ndim=2, nsig=2, f_signal=sigfunc, sigma=sigma,
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi = ws,
        init_weight_values_tilt = wts,
        dtype=torch.float64,
    )
    t = torch.tensor([t], dtype=torch.float64)
    x = torch.tensor([x], dtype=torch.float64, requires_grad=True)
    dw = torch.tensor([dw], dtype=torch.float64)
    x_act = model.step(t, x, sigparams, dt, dw).detach().numpy()
    assert np.allclose(x_act, x_exp)
    
@pytest.mark.parametrize('ws, wts, sigparams, t, dt, tfin, \
                         sigma, dw, y, shape_exp', [
    [[W1, W2, W3], [WT1], [[5, 0, 1, 1, -1]], [0],  1e-2, [10],
     1e-3, [[[0.05, -0.024]]], [[[0, 1]]], (1,1,2)
    ],
    [[W1, W2, W3], [WT1], [[5, 0, 1, 1, -1]], [10], 1e-2, [10],
     1e-3, [[[0.05, -0.024]]], [[[0, 1]]], (1,1,2)
    ],
    [[W1, W2, W3], [WT1], [[5, 0, 1, 1, -1]], [0], 1e-2,  [10],
     1e-3, [[[0.05, -0.024],[0.05, -0.024]]], [[[0, 1], [0, 1]]], (1,2,2)
    ],
])
def test_simulate_forward(ws, wts, sigparams, t, dt, tfin, sigma, dw, y, shape_exp):
    ncells = len(y)
    sigfunc = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
    model = PhiNN(
        ndim=2, nsig=2, f_signal=sigfunc, sigma=sigma, ncells=ncells,
        testing=True, include_bias=False, 
        init_weights=True,
        init_weight_values_phi = ws,
        init_weight_values_tilt = wts,
        testing_dw=torch.tensor(dw, dtype=torch.float64),
        dtype=torch.float64,
    )
    # y = np.array(y)
    # x = np.concatenate([[t], [tfin], y.flatten(), [tcrit, *p0, *p1]])
    # x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    # y_act = model.simulate_forward(x, dt=dt).detach().numpy()

    y = np.array(y)
    sigparams = np.array(sigparams)
    # x = np.concatenate([[t], [tfin], y.flatten(), [tcrit, *p0, *p1]])
    # x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    t0 = torch.tensor(t, dtype=torch.float64)
    t1 = torch.tensor(tfin, dtype=torch.float64)
    y0 = torch.tensor(y, dtype=torch.float64)
    y0.requires_grad_()
    sigparams = torch.tensor(sigparams, dtype=torch.float64)
    y_act = model.simulate_forward(t0, t1, y0, sigparams, dt=dt).detach().numpy()

    assert y_act.shape == shape_exp
    

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


@pytest.mark.parametrize('ncells, ws, wts, sigparams, t0, dt, tfin, \
                         sigma, dw, y0, y1, yfin_exp', [
    [4, [W1, W2, W3], [WT1], 
     [[5, 0, 1, 1, -1]], 
     [0], 1e-3, [2e-3],
     0, 
     [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], 
     [[[0, 0], [0, 1], [1, 0], [1, 1]]],
     [[[0, 0], [0, 1], [1, 0], [1, 1]]],
     [[[-0.0199185, -0.0299297], 
       [-0.000878827, 0.999681], 
       [0.989734, -0.00749453], 
       [0.992008, 0.997977]]],
    ],

])
class TestNoNoise:

    def _get_x(self, t0, tfin, y0, sigparams):
        nbatches = len(t0)
        t0 = np.array([t0])
        tfin = np.array([tfin])
        y0 = np.array(y0).reshape([nbatches, -1])
        sigparams = np.array(sigparams)
        x = np.concatenate([t0, tfin, y0, sigparams], axis=1)
        return x

    def test_forward(self, ncells, ws, wts, sigparams, t0, dt, tfin, 
                              sigma, dw, y0, y1, yfin_exp):
        sigfunc = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
        model = PhiNN(
            ndim=2, nsig=2, f_signal=sigfunc, sigma=sigma, ncells=ncells,
            testing=True, include_bias=False, 
            init_weights=True,
            init_weight_values_phi = ws,
            init_weight_values_tilt = wts,
            testing_dw=torch.tensor(dw, dtype=torch.float64),
            dtype=torch.float64,
        )
        x = self._get_x(t0, tfin, y0, sigparams)
        yfin_exp = np.array(yfin_exp)
        x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        yfin_act = model.forward(x, dt=dt).detach().numpy()
        assert np.allclose(yfin_exp, yfin_act), \
            f"Expected:\n {yfin_exp}\nGot:\n{yfin_act}"
        
    def test_loss(self, ncells, ws, wts, sigparams, t0, dt, tfin, 
                  sigma, dw, y0, y1, yfin_exp):
        sigfunc = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
        model = PhiNN(
            ndim=2, nsig=2, f_signal=sigfunc, sigma=sigma, ncells=ncells,
            testing=True, include_bias=False, 
            init_weights=True,
            init_weight_values_phi = ws,
            init_weight_values_tilt = wts,
            testing_dw=torch.tensor(dw, dtype=torch.float64),
            dtype=torch.float64,
        )
        x = self._get_x(t0, tfin, y0, sigparams)
        x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        yfin_act = model.forward(x, dt=dt)
        loss_act = mean_cov_loss(
            yfin_act, torch.tensor(y1, dtype=torch.float64)
        ).detach().numpy()
        mu_exp = np.mean(yfin_exp, 1)
        print(mu_exp)
        cov_exp = np.array([np.cov(np.array(yf).T) for yf in yfin_exp])
        print(cov_exp)
        mu_loss = np.array([np.sum(np.square(mu_exp - np.mean(y))) for y in y1])
        cov_loss = np.array([np.sum(np.square(cov_exp - np.cov(np.array(y).T))) for y in y1])
        loss_exp = mu_loss + cov_loss
        assert np.allclose(loss_exp, loss_act), \
            f"Expected:\n {loss_exp}\nGot:\n{loss_act}"
        
    def test_mean_loss(self, ncells, ws, wts, sigparams, t0, dt, tfin, 
                       sigma, dw, y0, y1, yfin_exp):
        sigfunc = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
        model = PhiNN(
            ndim=2, nsig=2, f_signal=sigfunc, sigma=sigma, ncells=ncells,
            testing=True, include_bias=False, 
            init_weights=True,
            init_weight_values_phi = ws,
            init_weight_values_tilt = wts,
            testing_dw=torch.tensor(dw, dtype=torch.float64),
            dtype=torch.float64,
        )
        y0 = np.array(y0)
        x = self._get_x(t0, tfin, y0, sigparams)
        x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        yfin_act = model.forward(x, dt=dt)
        loss_act = mean_diff_loss(
            yfin_act, torch.tensor(y1, dtype=torch.float64)
        ).detach().numpy()
        mu_exp = np.mean(yfin_exp, 1)
        print(mu_exp)
        loss_exp = np.sum(np.square(mu_exp - np.mean(y1, 1)))
        assert np.allclose(loss_exp, loss_act), \
            f"Expected:\n {loss_exp}\nGot:\n{loss_act}"
        
    @pytest.mark.parametrize('grad_w1_exp, grad_w2_exp, grad_w3_exp, grad_ws_exp', [
        [[[-0.000021717, 0.0000169943], 
          [0.0000546151, 0.0000572351], 
          [-0.000039655, 7.1173e-7]],
         [[0.00016531, 0.000145222, 0.000155356], 
          [0.00013069, 0.000113604, 0.000128925], 
          [0.0000354146, 0.0000352284, 0.0000377503]],
         [[-6.44939e-6, 0.0000415891, 0.000079741]],
         [[0, 0.000038971], [0, 0.0000397822]]
        ],
    ])
    def test_loss_gradient(self, ncells, ws, wts, sigparams, t0, dt, tfin, 
                              sigma, dw, y0, y1, yfin_exp, grad_w1_exp, 
                              grad_w2_exp, grad_w3_exp, grad_ws_exp):
        sigfunc = lambda t, p: jump_function(t, p[...,0], p[...,1:3], p[...,3:])
        model = PhiNN(
            ndim=2, nsig=2, f_signal=sigfunc, sigma=sigma, ncells=ncells,
            testing=True, include_bias=False, 
            init_weights=True,
            init_weight_values_phi = ws,
            init_weight_values_tilt = wts,
            testing_dw=torch.tensor(dw, dtype=torch.float64),
            dtype=torch.float64,
        )
        y0 = np.array(y0)
        x = self._get_x(t0, tfin, y0, sigparams)
        x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        yfin_act = model.forward(x, dt=dt)
        loss = mean_diff_loss(
            yfin_act, torch.tensor(y1, dtype=torch.float64)
        )
        loss.backward()
        plist = list(model.parameters())
        errors = []
        grads_exp = [grad_w1_exp, grad_w2_exp, grad_w3_exp, grad_ws_exp]
        for i, p in enumerate(plist):
            pgrad = p.grad.detach().numpy()
            pgrad_exp = grads_exp[i]
            if not np.allclose(pgrad, pgrad_exp):
                msg = f"Mismatch for grad W{i}."
                msg += f"Expected:\n{pgrad_exp}\nGot:\n{pgrad}"
                errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format('\n'.join(errors))
                