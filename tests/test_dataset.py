import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
from phiml.dataset import LandscapeSimulationDataset

#####################
##  Configuration  ##
#####################

# Configuration settings...

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

@pytest.mark.parametrize("datdir, nsims, dim, ts, xs_shape, ps_shape", [
    ["tests/data", 2, 5, np.linspace(0, 10, 1 + int(10 / 0.1)), (100,2), (5,)]
])
class TestDataset:
    
    def test_length(self, datdir, nsims, dim, ts, xs_shape, ps_shape):
        dataset = LandscapeSimulationDataset(datdir, nsims, dim)
        assert len(dataset) == nsims * (len(ts) - 1)

    def test_shapes(self, datdir, nsims, dim, ts, xs_shape, ps_shape):
        dataset = LandscapeSimulationDataset(datdir, nsims, dim)
        for data in dataset:
            (t0, x0, t1, p), x1 = data
            names = ['t0', 'x0', 't1', 'x1', 'p']
            shapes_act = [t0.shape, x0.shape, t1.shape, x1.shape, p.shape]
            shapes_exp = [(), xs_shape, (), xs_shape, ps_shape]
            errors = []
            for i, name in enumerate(names):
                shape_act = shapes_act[i]
                shape_exp = shapes_exp[i]
                if not (shape_act == shape_exp):
                    msg = f"Bad {name} shape. Expected {shape_exp}. Got {shape_act}."
                    errors.append(msg)
            assert not errors, "Errors occured:\n{'\n'.join(errors)}"
    

TRAINDIR = "tests/simtest1/data_train"
NSIMS_TRAIN = 4
OUTDIR = "tests/simtest1/tmp_out"
@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('batch_size', [1, 2])
class TestDataloader:

    def test_train_dataset(self, device, dtype, batch_size):
        train_dataset = LandscapeSimulationDataset(
            TRAINDIR, NSIMS_TRAIN, 2, 
            transform='tensor', 
            target_transform='tensor',
            dtype=dtype,
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=False,
        )
        assert len(train_dataset) == 4