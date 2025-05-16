import os
os.system('gcc -O3 -ffast-math -funroll-loops -mavx2 -march=native -c v1.c')
# os.system('gcc -O3 -c v1.c')
os.system('gcc -shared v1.o -o v1.so -lm')
os.system('ctypesgen v1.h -l v1.so > v1_ct.py')

import numpy as np

import v0
import v1_ct

weights_f = v0.sim.connectivity.weights.astype('f')
idelays_u32 = v0.sim.connectivity.idelays.astype(np.uint32)
K_bath_f = v0.sim.model.K_bath.astype('f')

import ctypes as ct

fp = ct.POINTER(ct.c_float)
up = ct.POINTER(ct.c_uint32)
wp = weights_f.ctypes.data_as(fp)
idp = idelays_u32.ctypes.data_as(up)
kp = K_bath_f.ctypes.data_as(fp)

states_f = v0.init_cond.astype('f')
statesp = states_f.ctypes.data_as(fp)

ntime = int(v0.sim.simulation_length/v0.sim.integrator.dt)
N = weights_f.shape[0]
raw_f = np.zeros((ntime, 5, N), 'f')
rawp = raw_f.ctypes.data_as(fp)

history = v0.sim.history.buffer[:, 0, :, 0].T.astype('f')
maxdelay = v0.sim.connectivity.idelays.max()
assert history.shape == (N, maxdelay+1)
historyp = history.ctypes.data_as(fp)

s = v1_ct.sim_t(
    nnode=N,
    nsvar=5,
    ntime=ntime,
    maxdelay=v0.sim.connectivity.idelays.max(),
    weights=wp,
    idelays=idp,
    dt=v0.sim.integrator.dt,
    G=v0.sim.coupling.a[0],
    K_bath=kp,
    progress_period=v0.sim.monitors[1].period,
    states=statesp,
    raw=rawp,
    history=historyp,
)
print(s, s.ntime)

sp = ct.POINTER(v1_ct.sim_t)(s)

if __name__ == '__main__':
    import time
    tik = time.time()
    v1_ct.sim_run(sp)
    tok = time.time()
    print(ntime / (tok - tik), 'iter/s')
    y0 = np.load('/tmp/v0.npy')
    import matplotlib as mpl
    mpl.use('qt5agg', force=True)
    import pylab as pl
    pl.figure(figsize=(10, 10))
    t = np.r_[:ntime]*s.dt
    pl.plot(t, raw_f[:, 0, :] + np.r_[:N]*2, 'k', alpha=0.7)
    pl.plot(t, y0[:, 0, :, 0] + np.r_[:N]*2, 'r', alpha=0.7)
    pl.show()
    pl.savefig('v1.jpg')
    # np.testing.assert_allclose(raw_f[:,0], y0[:,0,:,0], 1e-3, 1e-3)
