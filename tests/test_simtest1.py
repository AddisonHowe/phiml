import pytest
import numpy as np
import torch
from model import PhiNN
from helpers import get_binary_function
from helpers import jump_function, mean_cov_loss, mean_diff_loss

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


TESTDIR = "tests/simtest1/data_train"

@pytest.mark.parametrize('device', ['cpu'])
def test_dataloader(device):
    pass
