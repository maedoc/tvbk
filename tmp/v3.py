import os
os.system('ispc --pic --math-lib=fast -O3 -h v3.h v3.ispc.c -o v3.o')
os.system('gcc -shared v3.o -o v3.so -lm')
os.system('ctypesgen v3.h -l v3.so > v3_ct.py')

import numpy as np

import v0
import v3_ct

weights_f = np.ascontiguousarray(v0.sim.connectivity.weights.T).astype('f')
N = weights_f.shape[0]
idelays_u32 = np.ascontiguousarray(v0.sim.connectivity.idelays.T).astype(np.uint32)
print(weights_f.strides)
K_bath_f = v0.sim.model.K_bath.astype('f')

import ctypes as ct

fp = ct.POINTER(ct.c_float)
up = ct.POINTER(ct.c_uint32)
wp = weights_f.ctypes.data_as(fp)
idp = idelays_u32.ctypes.data_as(up)
kp = K_bath_f.ctypes.data_as(fp)

assert v0.init_cond.shape == (1, 5, N, 1)
states_f = v0.init_cond[0, :, :, 0].astype('f')
statesp = states_f.ctypes.data_as(fp)

buf_f = v0.sim.history.buffer[:, 0, :, 0].T.copy().astype('f')  # N,horizon
H = idelays_u32.max() + 1
assert buf_f.shape == (N, H)
H2 = 2**int(np.ceil(np.log2(H)))
assert H2 == 1024
buf_f2 = np.hstack([buf_f, np.zeros((N, H-H2), 'f')])
assert buf_f2.shape == (N, H2)
buf_p = buf_f2.ctypes.data_as(fp)

ntime = int(v0.sim.simulation_length/v0.sim.integrator.dt)
raw_f = np.zeros((ntime, 5, N), 'f')
rawp = raw_f.ctypes.data_as(fp)

cx_f = np.zeros((ntime, N), 'f')
cx_p = cx_f.ctypes.data_as(fp)

s = v3_ct.sim_t(
    nnode=N,
    nsvar=5,
    ntime=ntime,
    h2=H2,
    maxdelay=v0.sim.connectivity.idelays.max(),
    weights=wp,
    idelays=idp,
    dt=v0.sim.integrator.dt,
    G=v0.sim.coupling.a[0],
    K_bath=kp,
    progress_period=v0.sim.monitors[1].period,
    states=statesp,
    raw=rawp,
    history=buf_p,
    cx=cx_p,
)

sp = ct.POINTER(v3_ct.sim_t)(s)

if __name__ == '__main__':
    import time
    tik = time.time()
    v3_ct.sim_run(sp)
    tok = time.time()
    print(ntime / (tok - tik), 'iter/s')

    y0 = np.load('/tmp/v0.npy')
    cx0 = np.load('/tmp/v0-cx.npy')
    import matplotlib as mpl
    mpl.use('qt5agg', force=True)
    import pylab as pl
    pl.figure(figsize=(10, 10))
    t = np.r_[:ntime]*s.dt
    pl.plot(t[::5], raw_f[::5, 0, :] + np.r_[:N]*2, 'k', alpha=0.7)
    pl.plot(t[::5], y0[::5, 0, :, 0] + np.r_[:N]*2, 'r', alpha=0.7)
    # pl.plot(t[::5], cx_f[::5] + np.r_[:N]*2, 'k', alpha=0.7)
    # pl.plot(t[::5], cx0[::5] + np.r_[:N]*2, 'r', alpha=0.7)
    pl.show()
    pl.savefig('v3.jpg')
    # np.testing.assert_allclose(raw_f[:,0], y0[:,0,:,0], 1e-3, 1e-3)
    print('all done')
