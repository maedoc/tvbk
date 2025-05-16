import io
import contextlib

import numpy as np
import pickle
import torch
from sbi.inference import NPE_A as NPE


def to_torch(x):
    import torch
    return torch.from_numpy(np.array(x)).float()


def run_sbi(theta, features, fname=None, prog=True):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        mu = to_torch(np.mean(theta, axis=0))
        cov = to_torch(np.cov(theta.T))
        prior = torch.distributions.MultivariateNormal(mu, cov)
        inference = NPE(prior=prior, show_progress_bars=prog)
        inference.append_simulations(to_torch(theta), to_torch(features))
        inference.train()
        posterior = inference.build_posterior()
    if prog:
        print(out.getvalue())
    if fname:
        with open(fname, 'wb') as fd:
            pickle.dump(posterior, fd)
    return posterior

def uniform_var(a,b):
    return (b - a)**2 / 12.

def posterior_diags(p_us, po_us, true_us):
    po_u = np.mean(po_us.numpy(), axis=0)
    po_sd = np.std(po_us.numpy(), axis=0)
    po_z = np.abs((po_u - true_us)/po_sd)
    p_var = np.var(p_us, axis=0) if hasattr(p_us, 'size') else uniform_var(*p_us)
    po_shrink = np.array(1 - po_sd**2/p_var)
    # check true in 90% ci
    q5, q95 = np.quantile(po_us, [0.05, 0.95], axis=0)
    ci90 = np.array((q5 < true_us) * (true_us < q95))
    return po_shrink, po_z, ci90

