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
