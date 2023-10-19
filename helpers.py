"""General Helper Functions
"""

import torch

def get_binary_function(tcrit, p0, p1):
    """Return a binary function that changes from p0 to p1 at time tcrit."""
    return lambda t: p0 if t < tcrit else p1

def jump_function(t, tcrit, p0, p1):
    return (t < tcrit) * p0 + (t >= tcrit) * p1

def select_device(overwrite=None):
    if overwrite:
        return overwrite
    if torch.cuda.is_available():
        return 'cuda'
    try:
        if torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    except AttributeError as e:
        return 'cpu'

def mean_cov_loss(y_sim, y_obs):
    mu_sim = torch.mean(y_sim, dim=-2)
    mu_obs = torch.mean(y_obs, dim=-2)
    cov_sim = batch_cov(y_sim)
    cov_obs = batch_cov(y_obs)
    mu_err = torch.sum(torch.square(mu_sim - mu_obs), axis=1)
    cov_err = torch.sum(torch.square(cov_sim - cov_obs), dim=[1,2])
    return torch.mean(mu_err + cov_err)

def batch_cov(batch_points):
    """
    Returns:
        Shape (b,d,d) tensor
    """
    b, n, d = batch_points.size()
    mean = batch_points.mean(dim=1).unsqueeze(1)
    diffs = (batch_points - mean).reshape(b*n, d)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(b,n,d,d)
    bcov = prods.sum(dim=1) / (n - 1)  # unbiased estimate
    return bcov