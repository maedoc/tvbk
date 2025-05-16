from sbi_util import to_torch, run_sbi, posterior_diags
import matplotlib as mpl
mpl.use('qt5agg', force=True)
import numpy as np
import pylab as pl

# tavg x and thetas, shaped nbatch,8
print('load batched parameters & tavg')
x = np.lib.format.open_memmap('/tmp/v9-x.npy')
theta = np.load('/tmp/v9-theta.npy')
nbatch, _, ntime, nn = x.shape
nsims = nbatch * 8

print('compute envelope')
ex = []
for i in range(x.shape[0]):
    ex.append(
        np.abs(x[i]).reshape(8, 50, -1, nn).mean(axis=2))
ex = np.array(ex, dtype='f').reshape(nsims, 50, nn)

# but reshape for ease
assert theta.shape == (nbatch, 8, 1+nn)
theta = theta.reshape(nsims, 1+nn)

# cut off beginning where nothing happens
mex = ex.max(axis=-1).max(axis=0)
t0 = np.argwhere(mex > 0.1)[0, 0]
ex = ex[:, t0:]

debug = False
if debug:
    # show envelope for a few sims
    for i in range(25):
        pl.subplot(5,5,i+1)
        pl.imshow(ex[i].T)
        pl.title(f'G[{i}] = {theta[i,0]:0.2f}')
    pl.tight_layout()
    pl.show()

# format envelope as data feature
ex = ex.reshape(nsims, -1)
print('run sbi, theta & data features shaped', theta.shape, ex.shape)

# now train an sbi estimator on them
posterior = run_sbi(theta, ex)

# and test on one simulation
theta0 = theta[0]
import time
tik = time.time()
theta0_hat = posterior.sample((200, ), x=ex[0])
tok = time.time()
print('sampling took', tok-tik, 's')
ps, pz, ci = posterior_diags(
    p_us=theta,  # take as prior full theta samples
    po_us=theta0_hat,
    true_us=theta0
)
print(f'shrink {ps.mean():0.2f}, z {pz.mean():0.2f}, ci {ci.mean():0.2f}')
