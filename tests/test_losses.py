import pytest
import numpy as np
import torch
from helpers import mean_cov_loss, mean_diff_loss, kl_divergence_est

@pytest.mark.parametrize('dtype, atol', [
    [torch.float32, 1e-6], 
    [torch.float64, 1e-6]
])
@pytest.mark.parametrize('xpath, ypath, loss_exp', [
    ['tests/loss_test_data/xtest1.npy', 
     'tests/loss_test_data/ytest1.npy',
     3.061594942272676],
    ['tests/loss_test_data/ytest1.npy', 
     'tests/loss_test_data/xtest1.npy',
     2.5479272873648875],
    ['tests/loss_test_data/xtest2.npy', 
     'tests/loss_test_data/ytest2.npy',
     0.8700861617889799],
])
def test_kl_loss(dtype, atol, xpath, ypath, loss_exp):
    x = torch.tensor(np.load(xpath), dtype=dtype)
    y = torch.tensor(np.load(ypath), dtype=dtype)
    loss_act = kl_divergence_est(x, y).numpy()
    assert np.allclose(loss_exp, loss_act, atol=atol), \
        f"Expected:\n{loss_exp}\nGot:\n{loss_act}"
    