import os
os.system('gcc -O3 -ffast-math -funroll-loops -mavx2 -march=native -fopenmp-simd -c v2.c')
# os.system('gcc -O3 -c v2.c')
os.system('gcc -shared v2.o -o v2.so -lm')
os.system('ctypesgen v1.h -l v2.so > v2_ct.py')

import numpy as np

import v0
import v2_ct

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
raw_f = np.zeros((ntime, 5, weights_f.shape[0]), 'f')
rawp = raw_f.ctypes.data_as(fp)

s = v2_ct.sim_t(
    nnode=weights_f.shape[0],
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
)
print(s, s.ntime)

sp = ct.POINTER(v2_ct.sim_t)(s)

import time
tik = time.time()
v2_ct.sim_run(sp)
tok = time.time()
print(ntime / (tok - tik), 'iter/s')
