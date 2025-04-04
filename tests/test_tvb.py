import pytest
import numpy as np
import scipy.sparse

import tvb.simulator.lab as tvb
import tvb.simulator.backend.nb_mpr as nb_mpr
import tvbk


def nextpow2(i):
    return int(2**np.ceil(np.log2(i)))

def tvbk_run_sim(sim, init_sim_state):
    conn = sim.connectivity
    s_w = scipy.sparse.csr_matrix(conn.weights)
    nr = conn.weights.shape[0]
    k_cx = tvbk.Cx8s(nr, nextpow2(conn.horizon), 1)
    k_conn = tvbk.Conn(nr, s_w.data.size)
    k_conn.weights[:] = s_w.data.astype(np.float32)
    k_conn.indptr[:] = s_w.indptr.astype(np.uint32)
    k_conn.indices[:] = s_w.indices.astype(np.uint32)
    k_conn.idelays[:] = (
        conn.tract_lengths[conn.weights != 0] / conn.speed / sim.integrator.dt
    ).astype(np.uint32)
    assert k_conn.idelays.min() >= 2
    # print(k_cx.buf.shape) # (batch, 90, horizon2, 8)
    # print(sim.history.buffer.shape)  # (horizon, 2, 90, 1)
    k_cx.buf[:] = 0.
    k_cx.buf[0, :, -conn.horizon:] = sim.history.buffer[:, 0, :, 0].T[..., None]
    num_svar, num_parm = 2, 6
    x = np.zeros((1, num_svar, nr, 8), 'f')
    p = np.zeros((1, nr, num_parm, 8), 'f')
    x[:] = init_sim_state  # sim.current_state[:, :, 0][..., None]
    for i, pname in enumerate('tau,I,Delta,J,eta,cr'.split(',')):
        p[:, :, i] = getattr(sim.model, pname)
        print(pname, p[0, 0, i, 0])
    num_time = int(sim.simulation_length / sim.integrator.dt)
    num_skip = int(sim.monitors[0].period / sim.integrator.dt)
    # assert num_skip == 10
    y2 = np.zeros((num_time // num_skip, 2, nr), 'f')
    k_y = np.zeros_like(x)
    z = np.zeros((1, num_svar, 8), 'f')
    seed = np.zeros((1, 8, 4), np.uint64)
    for t0 in range(y2.shape[0]):
        tvbk.step_mpr(k_cx, k_conn, x, k_y, z, p,
                      t0*num_skip, num_skip, sim.integrator.dt,
                      seed)
        y2[t0] = k_y[..., 0]
    return y2


def make_tvb_model(simlen=1e2, dt=0.1, period=1.0):
    model = tvb.models.MontbrioPazoRoxin(tau=np.r_[10.0])
    conn = tvb.connectivity.Connectivity()
    nn = 90
    conn.centres_spherical(number_of_regions=nn)
    conn.motif_chain_directed(number_of_regions=nn)
    conn.tract_lengths += 3*dt
    noise = tvb.noise.Additive(nsig=np.r_[0., 0.])
    # conn.configure()
    sim = tvb.simulator.Simulator(
        model=model,
        connectivity=conn,
        monitors=[tvb.monitors.TemporalAverage(period=period)],
        integrator=tvb.integrators.HeunStochastic(dt=dt, noise=noise),
        simulation_length=simlen,
    ).configure()
    return sim
    

def test_tvb1():
    sim = make_tvb_model()
    init_sim_state = sim.current_state.copy()
    (t1, y1), = nb_mpr.NbMPRBackend().run_sim(sim, chunksize=100)
    (t0, y0), = sim.run()
    y2 = tvbk_run_sim(sim, init_sim_state)
    for t in range(100):
        np.testing.assert_allclose(y0[t, 1, :, 0], y2[t, 1], 0.01, 0.3)

perf_time = 1e4
perf_period = 1.0

# so slow
# @pytest.mark.benchmark(group='mpr1')
# def test_tvb_perf(benchmark):
#     sim = make_tvb_model(perf_time)
#     benchmark(sim.run)

@pytest.mark.benchmark(group='mpr1')
def test_nbmpr_perf(benchmark):
    sim = make_tvb_model(perf_time, perf_period)
    benchmark(lambda : nb_mpr.NbMPRBackend().run_sim(sim, chunksize=1000))

@pytest.mark.benchmark(group='mpr1')
def test_tvbk_perf(benchmark):
    sim = make_tvb_model(perf_time, perf_period)
    init_state = sim.current_state.copy()
    benchmark(lambda : tvbk_run_sim(sim, init_state))

def test_kionex():
    from tvb.simulator.models import KIonEx
    model = KIonEx()
    for i in range(1024):
        dx = np.zeros((5, 8), 'f')
        x = np.random.randn(*dx.shape).astype('f')/5 + np.c_[0.1, -50, 0.5, -5, -10].T
        c = np.random.randn(1,8).astype('f')/2
        # Parameters in same order as kionex.hpp
        p = np.array([
            0.0,    # E
            5.5,    # K_bath
            0.1,    # J
            0.0,    # eta
            1.0,    # Delta
            -40.0,  # c_minus
            0.5,    # R_minus
            -20.0,  # c_plus
            -0.5,   # R_plus
            -31.0,  # Vstar
            1.0,    # Cm
            4.0,    # tau_n
            0.04,   # gamma
            0.001   # epsilon
        ], dtype='f').reshape(14, 1).repeat(8, axis=1)
        assert p.shape == (16, 8)
        m.dfun_kionex8(dx, x, c, p)
        # models imported from TVB expect a 3rd dim which can just be 1
        dx_np = model.dfun(x[:,:,None], c[:,:,None], 0)[:,:,0]
        np.testing.assert_allclose(dx, dx_np, 0.15, 0.1)

