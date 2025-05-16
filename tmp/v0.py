import tvb.simulator.lab as tvb
from tvb.simulator.models.base import Model
from tvb.basic.neotraits.api import NArray, List, Range, Final
import numpy as np


cx = []
class KIonEx(tvb.simulator.models.KIonEx):
    # Sourin's modif just adds J*r*(E - V) to dV/dt, so we reuse the TVB impl here
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        cx.append(coupling)
        derivative = super().dfun(state_variables, coupling, local_coupling)
        # if_Vdot = Vdot - R_minus*x**2 + eta + J*r*(E-V) + (R_minus/numpy.pi)*Coupling_Term*(E-V)
        # else_Vdot = Vdot - R_plus*x**2 + eta + J*r*(E-V) + (R_minus/numpy.pi)*Coupling_Term*(E-V)
        x, V, *_ = state_variables
        r = self.R_minus*x/np.pi
        derivative[1] += self.J*r*(self.E - V)
        return derivative


conn = tvb.connectivity.Connectivity.from_file()  # load default 76 regions
conn.weights = conn.weights / conn.weights.max()
# conn.tract_lengths[:] = 5.0
conn.speed = np.r_[15.0]
N = conn.weights.shape[0]

K_bath = np.ones(N) * 5.5
epi_reg = 42
K_bath[epi_reg] = 15.5
prop_reg = 21
K_bath[prop_reg] = 12.5


# random but consistent initial conditions
np.random.seed(42)
x_0 = np.random.uniform(low=0.1, high=0.1, size=((1, 1, N, 1)))
V_0 = np.random.uniform(low=-15., high=-15., size=((1, 1, N, 1)))
n_0 = np.random.uniform(low=0.45, high=0.45, size=((1, 1, N, 1)))
DKi_0 = np.random.uniform(low=0.2, high=0.2, size=((1, 1, N, 1)))
Kg_0 = np.random.uniform(low=-1., high=-1., size=((1, 1, N, 1)))
init_cond = np.concatenate([x_0, V_0, n_0, DKi_0, Kg_0], axis=1)

sim = tvb.simulator.Simulator(
    connectivity=conn,
    model=KIonEx(K_bath=K_bath),
    coupling = tvb.coupling.Scaling(a=np.r_[1.0]),
    integrator=tvb.integrators.HeunDeterministic(dt=0.01),
    monitors=[
        tvb.monitors.Raw(),
        tvb.monitors.ProgressLogger(period=200.0),
    ],
    simulation_length=3e3,
    initial_conditions=init_cond,
).configure()

if __name__ == '__main__':
    # only run the configured simulation if this is the main script

    import time
    tik = time.time()
    (t, y), *_ = sim.run() 
    tok = time.time()
    print(t.size / (tok-tik), 'iter/s')

    import pylab as pl
    pl.figure(figsize=(10, 10))
    pl.plot(t, y[:, 0, :, 0] + np.r_[:N]*2, 'k', alpha=0.7)
    pl.savefig('v0.jpg')

    np.save('/tmp/v0.npy', y)
    np.save('/tmp/v0-cx.npy', np.array(cx)[::2, 0, :, 0])


