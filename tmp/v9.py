# this is just v8 w/ some helpers for sbi
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import v8

wanted_nsims = 2048
x_save_path = '/tmp/v9-x.npy'
theta_save_path = '/tmp/v9-theta.npy'

batch_size = 8  # XXX do not change this
nbatch = wanted_nsims // batch_size
nsims = nbatch * batch_size

# sample parameter space for nsims
np.random.seed(42)
Gs = np.random.uniform(low=0., high=1.0, size=(nsims, 1))
Kbaths = np.random.uniform(low=5., high=15., size=(nsims, v8.N))
thetas = np.hstack([Gs, Kbaths])  # nsims, 1+N
batched_thetas = thetas.reshape(nbatch, 8, thetas.shape[-1])
np.save(theta_save_path, batched_thetas)

# create output file
save_x = np.lib.format.open_memmap(
    x_save_path, mode='w+', dtype='f',
    shape=(nbatch, 8, v8.ntavg, v8.N))

# helper function to build fast sim and apply to our parameters
def run1batch(i_batch, thetas):
    Gs = np.ascontiguousarray(thetas[:, 0])
    Kbaths = np.ascontiguousarray(thetas[:, 1:].T)
    assert Gs.shape == (8,)
    assert Kbaths.shape == (v8.N, 8)
    s = v8.make_sim(Gs, Kbaths)
    v8.v8_ct.sim_run(s['sp'])
    tavg_x = s['tavg_f'][:, 0].transpose(2, 0, 1)
    assert tavg_x.shape == (8, v8.ntavg, v8.N)
    save_x[i_batch] = tavg_x

pbar = tqdm.tqdm(total=nbatch)
with ThreadPoolExecutor() as exe:
    futures = [exe.submit(run1batch, i_batch, bt)
               for i_batch, bt in enumerate(batched_thetas)]
    for f in as_completed(futures):
        pbar.update(n=1)
        f.result()
