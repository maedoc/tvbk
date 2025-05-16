# drop raw & cx for tavg
import os
if os.path.getmtime('v8_ct.py') < os.path.getmtime('v8.ispc.c'):
    os.system('ispc -g --pic --math-lib=fast -O3 -h v8.h v8.ispc.c -o v8.o')
    os.system('gcc -shared v8.o -o _v8.so -lm')
    os.system('ctypesgen v8.h -l _v8.so > v8_ct.py')

import numpy as np
import v0
import v8_ct

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
ntime = int(v0.sim.simulation_length/v0.sim.integrator.dt)
ntavg = ntime // 10

def make_sim(Gs, Kbaths):
    assert v0.init_cond.shape == (1, 5, N, 1)
    states_f = np.zeros((5, N, 8), 'f')
    states_f += v0.init_cond[0].astype('f')
    statesp = states_f.ctypes.data_as(fp)
    buf_f = v0.sim.history.buffer[:, 0, :, 0].T.copy().astype('f')  # N,horizon
    H = idelays_u32.max() + 1
    H2 = 2**int(np.ceil(np.log2(H)))
    buf_f2 = np.hstack([np.zeros((N, H-H2), 'f'), buf_f])
    assert buf_f2.shape == (N, H2)
    buf_f2_8 = np.zeros(buf_f2.shape + (8,), 'f')
    buf_f2_8 += buf_f2[..., None]
    buf_p = buf_f2_8.ctypes.data_as(fp)
    tavg_f = np.zeros((ntavg, 5, N, 8), 'f')
    tavgp = tavg_f.ctypes.data_as(fp)
    G_f = Gs.astype('f')
    G_p = G_f.ctypes.data_as(fp)
    K_bath_f = Kbaths.astype('f')
    K_bath_p = K_bath_f.ctypes.data_as(fp)
    s = v8_ct.sim_t(
        nnode=N,
        nsvar=5,
        ntime=ntime,
        h2=H2,
        maxdelay=v0.sim.connectivity.idelays.max(),
        weights=wp,
        idelays=idp,
        dt=v0.sim.integrator.dt,
        G=G_p,
        K_bath=K_bath_p,
        progress_period=1e4,  # v0.sim.monitors[1].period,
        states=statesp,
        history=buf_p,
        tavg=tavgp,
        ntavg=ntavg
    )
    sp = ct.POINTER(v8_ct.sim_t)(s)
    return locals()


if __name__ == '__main__':
    import concurrent.futures
    import time
    tik = time.time()
    Gs = np.zeros((8,)) + v0.sim.coupling.a[0]
    Kbaths = np.zeros((N, 8)) + v0.sim.model.K_bath.reshape(-1, 1)
    if False:
        for i in range(100):
            ss = [make_sim(Gs, Kbaths) for _ in range(128)]
            tik = time.time()
            with concurrent.futures.ThreadPoolExecutor() as exe:
                exe.map(v8_ct.sim_run, [s['sp'] for s in ss])
            tok = time.time()
            print(i, len(ss) * ntime * 8 / (tok - tik), 'iter/s')

    else:
        s = make_sim(Gs, Kbaths)
        v8_ct.sim_run(s['sp'])
        tok = time.time()
        print(ntime*8 / (tok - tik), 'iter/s')

        for i in range(7):
            np.testing.assert_allclose(
                s['tavg_f'][..., i+1],
                s['tavg_f'][..., 0]
            )

        import matplotlib as mpl
        mpl.use('qt5agg', force=True)
        import pylab as pl
        pl.figure(figsize=(10, 10))
        pl.plot(s['tavg_f'][:, 0, :, 0] + np.r_[:N]*2, 'k', alpha=0.7)
        pl.show()
        pl.savefig('v8.jpg')
