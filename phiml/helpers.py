"""General Helper Functions
"""

import numpy as np
import torch

def get_binary_function(tcrit, p0, p1):
    """Return a binary function that changes from p0 to p1 at time tcrit."""
    return lambda t: p0 if t < tcrit else p1

def jump_function(t, tcrit, p0, p1):
    return torch.reshape((t < tcrit), [t.shape[0],1]) * p0 + \
           torch.reshape((t >= tcrit), [t.shape[0],1]) * p1

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

def mean_diff_loss(y_sim, y_obs):
    mu_sim = torch.mean(y_sim, dim=-2)
    mu_obs = torch.mean(y_obs, dim=-2)
    mu_err = torch.sum(torch.square(mu_sim - mu_obs), axis=1)
    return torch.mean(mu_err)

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

def disp_mem_usage(msg=""):
    memalloc = torch.cuda.memory_allocated()
    maxmemalloc =torch.cuda.max_memory_allocated()
    print(f"[{msg}] mem: {memalloc}  \t  max: {maxmemalloc}", flush=True)

def kl_divergence_est(q_samp, p_samp):
    """Estimate the KL divergence. Returns the average over all batches.
    Adapted from:
      https://gist.github.com/atabakd/ed0f7581f8510c8587bc2f41a094b518

    Args:
        q_samp : Estimated sample distribution of shape (b,m,d)
        p_samp : Target sample distribution of shape (b,n,d)
    Returns:
        (float) KL estimate D(p|q), averaged over each batch.
    """

    _, n, d = p_samp.shape
    _, m, _ = q_samp.shape
    
    diffs_xx = torch.cdist(p_samp, p_samp, p=2, 
                           compute_mode='donot_use_mm_for_euclid_dist')  
    diffs_xy = torch.cdist(q_samp, p_samp, p=2, 
                           compute_mode='donot_use_mm_for_euclid_dist')
    
    r = torch.kthvalue(diffs_xx, 2, dim=1)[0]
    s = torch.kthvalue(diffs_xy, 1, dim=1)[0]

    vals = -torch.log(r/s).sum(axis=1) * d/n + np.log(m/(n-1.))
    return torch.mean(vals)
