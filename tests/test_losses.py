import pytest
import numpy as np
import torch
from helpers import mean_cov_loss, mean_diff_loss, kl_divergence_est

@pytest.mark.parametrize('xpath, ypath, loss_exp', [
    ['tests/loss_test_data/xtest1.npy', 
     'tests/loss_test_data/ytest1.npy',
     2.5479272873648875],
])
def test_kl_loss(xpath, ypath, loss_exp):
    x = torch.tensor(np.load(xpath))
    y = torch.tensor(np.load(ypath))
    loss_act = kl_divergence_est(x, y).numpy()
    assert np.allclose(loss_exp, loss_act), \
        f"Expected:\n{loss_exp}\nGot:\n{loss_act}"
    