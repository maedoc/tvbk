import pytest
pytestmark = pytest.mark.tvb

import numpy as np
import scipy.sparse
import tvbk as m
import tvbk

# Copied from ../../Nextcloud/2025/zz-sourin-kionex/Biophysical_wholebrain.md
# with minor adjustments for standalone use (e.g. Model base class)

# A minimal Model base class to allow KIonEx2 to be defined
import numpy
import tvb.simulator.lab as tvb
from tvb.basic.neotraits.api import NArray, List, Range, Final

class KIonEx2(tvb.models.Model):
    r"""
    KIonEx (Potassium K+ Ion exchange) mean-field model was developed in (Bandyopadhyay & Rabuffo et al. 2023). 
    It describes the mean-field activity of a population of Hodgkin-Huxley-type neurons (Depannemaker et al 2022) 
    linking the slow fluctuations of intra- and extra-cellular potassium ion concentrations to the mean membrane potential, 
    and the synaptic input to the population firing rate. 
    The model is derived as the mathematical limit of an infinite number of all-to-all coupled neurons, resulting in 5 state variables:
    :math:`x` represents a phenomenological variable connected to the firing rate, 
    :math:`V` represent the average membrane potential,
    :math:`n` represents the gating variable for potassium K, 
    :math:`\Delta K_{int}` represent the intracellular potassium concentration,
    :math:`K_g` represents the extracellular potassium buffering by the external bath
    """
    #_ui_name = "KIonEx"
    #ui_configurable_parameters = ['E', 'K_bath', 'J', 'eta', 'Delta','c_minus','R_minus','c_plus','R_plus','Vstar']

    E = NArray(
        label=r":math:`E`",
        default=numpy.array([0.]),
        domain=Range(lo=-80, hi=0, step=0.5),
        doc="""Reversal Potential""",
    )

    K_bath = NArray(
        label=r":math:`K_bath`",
        default=numpy.array([5.5]),
        domain=Range(lo=3, hi=40.0, step=0.25),
        doc="""Potassium concentration in bath""",
    )

    J = NArray(
        label=r":math:`J`",
        default=numpy.array([0.1]),
        domain=Range(lo=0.001, hi=40.0, step=0.01),
        doc="""Mean Synaptic weight""",
    )

    eta = NArray(
        label=":math:`eta`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.1),
        doc="""Mean heterogeneous noise""",
    )

    Delta = NArray(
        label=r":math:`\Delta`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""HWHM heterogeneous noise""",
    )

    c_minus = NArray(
        label=":math:`c_minus`",
        default=numpy.array([-40.0]),
        domain=Range(lo=-100.0, hi=-10.0, step=0.5),
        doc="""x-coordinate left parabola""",
    )

    R_minus = NArray(
        label=r":math:`R_minus`",
        default=numpy.array([0.5]),
        domain=Range(lo=0.0001, hi=5.0, step=0.01),
        doc="""curvature left parabola""",
    )

    c_plus = NArray(
        label=":math:`c_plus`",
        default=numpy.array([-20.0]),
        domain=Range(lo=-80.0, hi=0.0, step=0.5),
        doc="""x-coordinate right parabola""",
    )

    R_plus = NArray(
        label=r":math:`R_plus`",
        default=numpy.array([-0.5]),
        domain=Range(lo=-5.0, hi=-0.0001, step=0.01),
        doc="""curvature right parabola""",
    )


    Vstar = NArray(
        label=r":math:`Vstar`",
        default=numpy.array([-31]),
        domain=Range(lo=-55.0, hi=-15, step=0.5),
        doc="""x-coordinate meeting point of parabolas""",
    )

    #'Cm': 1, #nF, # membrane capacitance
    Cm = NArray(
        label=r":math:`Cm`",
        default=numpy.array([1]),
        domain=Range(lo=0.5, hi=1.5, step=0.1),
        doc="""membrane capacitance""",
    )

    # 'tau_n': 4, # ms # time constant of gating variable
    tau_n = NArray(
        label=r":math:`tau_n`",
        default=numpy.array([4]),
        domain=Range(lo=2, hi=6, step=0.5),
        doc="""time constant of gating variable""",
    )

    # 'gamma': 0.04,  # mol / C  # conversion factor
    gamma = NArray(
        label=r":math:`gamma`",
        default=numpy.array([0.04]),
        domain=Range(lo=0.02, hi=0.06, step=0.005),
        doc="""conversion factor""",
    )

    #'epsilon': 0.001, # mHz  # diffusion rate
    epsilon = NArray(
        label=r":math:`epsilon`",
        default=numpy.array([0.001]),
        domain=Range(lo=0.0005, hi=0.0015, step=0.0001),
        doc="""diffusion rate""",
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={
            "x": numpy.array([0.0, numpy.inf]),
            "V": numpy.array([-500.0, numpy.inf]),
            "n": numpy.array([0.0, numpy.inf]),
            "DKi": numpy.array([-100.0, numpy.inf]),
            "Kg": numpy.array([-100.0, numpy.inf])
        },
    )

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={
            "x": numpy.array([0., 1]),
            "V": numpy.array([-90., 10.]),
            "n": numpy.array([0., 1]),
            "DKi": numpy.array([-10, 0]),
            "Kg": numpy.array([-20, -5])
        },
        doc="""Expected ranges of the state variables for initial condition generation and phase plane setup.""",
    )



    # TODO should match cvars below..
    coupling_terms = Final(
        label="Coupling terms",
        # how to unpack coupling array
        default=["Coupling_Term"]
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("x", "V","n","DKi", "Kg"),
        default=("x", "V","n","DKi", "Kg"),
        doc="The quantities of interest for monitoring for the Infinite HH 5D.",
    )

    state_variables = ['x', 'V','n','DKi','Kg']
    _nvar = 5
    # Cvar is the coupling variable. 
    cvar = numpy.array([0], dtype=numpy.int32)
    # Stvar is the variable where stimulus is applied.
    stvar = numpy.array([1], dtype=numpy.int32)
    
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The mean-field approximation for a population of Hodgkin-Huxley-type neurons driven by slow potassium dynamics consists of a 5D system:

        .. math::
            \frac{dx}{dt}&=
            \begin{cases} 
            \Delta+2R_{-}(V-c_{-})x - J r x; \  V\leq V^{\star}\\
            \Delta+2R_{+}(V-c_{+})x - J r x; \  V> V^{\star},
            \end{cases}\\
            \frac{dV}{dt}&=
            \begin{cases} 
            -\frac{1}{C_m}(I_{Cl}+I_{Na}+I_{K}+I_{pump})-R_{-}x^2+J r(E_{syn}-V)+\overline{\eta}; \  V\leq V^{\star}\\
            -\frac{1}{C_m}(I_{Cl}+I_{Na}+I_{K}+I_{pump})-R_{+}x^2+J r(E_{syn}-V)+\overline{\eta}; \  V>V^{\star}, 
            \end{cases}\\
            \frac{dn}{dt} &= \frac{n_{\infty}(V)-n}{\tau_n}, \\
            \frac{d \Delta [K^{+}]_{int}}{dt} &= - \frac{\gamma}{\omega_i}(I_K - 2 I_{pump}),\\
            \frac{d[K^+]_g}{dt} &= \epsilon ([K^+]_{bath} - [K^+]_{ext}\}).\\

        For details refer to (Bandyopadhyay & Rabuffo et al. 2023)
        """
        
        x = state_variables[0, :]
        V = state_variables[1, :]
        n = state_variables[2, :]
        DKi = state_variables[3, :]
        Kg = state_variables[4, :]

        
        #[State_variables, nodes]
        E = self.E

        K_bath = self.K_bath
        J = self.J
        eta = self.eta
        Delta = self.Delta

        c_minus = self.c_minus
        R_minus = self.R_minus
        c_plus = self.c_plus
        R_plus = self.R_plus
        Vstar = self.Vstar

        Cm      = self.Cm
        tau_n   = self.tau_n
        gamma   = self.gamma
        epsilon = self.epsilon
     
        Coupling_Term = coupling[0, :] #This zero refers to the first element of cvar (trivial in this case)

        # Constants
        Cnap = 21.0  # mol.m**-3 
        DCnap = 2.0  # mol.m**-3 
        Ckp = 5.5  # mol.m**-3 
        DCkp = 1.0  # mol.m**-3 
        Cmna = -24.0  # mV 
        DCmna = 12.0  # mV 
        Chn = 0.4  # dimensionless 
        DChn = -8.0  # dimensionless 
        Cnk = -19.0  # mV 
        DCnk = 18.0  # mV #Ok in the paper
        g_Cl = 7.5  # nS #Ok in the paper   # chloride conductance
        g_Na = 40.0  # nS   # maximal sodiumconductance
        g_K = 22.0  # nS  # maximal potassium conductance
        g_Nal = 0.02  # nS  # sodium leak conductance
        g_Kl = 0.12  # nS  # potassium leak conductance
        rho = 250.  # 250.,#pA # maximal Na/K pump current
        w_i = 2160.0  # umeter**3  # intracellular volume 
        w_o = 720.0  # umeter**3 # extracellular volume 
        Na_i0 = 16.0  # mMol/m**3 # initial concentration of intracellular Na
        Na_o0 = 138.0 # mMol/m**3 # initial concentration of extracellular Na
        K_i0 = 130.0  # mMol/m**3 # initial concentration of intracellular K
        K_o0 = 4.80   # mMol/m**3 # initial concentration of extracellular K
        Cl_i0 = 5.0   # mMol/m**3 # initial concentration of intracellular Cl
        Cl_o0 = 112.0 # mMol/m**3 # initial concentration of extracellular Cl
        

        # helper functions

        def m_inf(V):
            return 1.0/(1.0+numpy.exp((Cmna-V)/DCmna))

        def n_inf(V):
            return 1.0/(1.0+numpy.exp((Cnk-V)/DCnk))

        def h(n):
            return 1.1 - 1.0 / (1.0 + numpy.exp(-8.0 * (n - 0.4)))

        def I_K_form(V,n,K_o,K_i):
            return (g_Kl+g_K*n)*(V- 26.64*numpy.log(K_o/K_i)) 

        def I_Na_form(V,Na_o,Na_i,n):
            return (g_Nal+g_Na*m_inf(V)*h(n))*(V- 26.64*(numpy.log(Na_o) - numpy.log(Na_i)))

        def I_Cl_form(V):
            return g_Cl*(V+ 26.64*numpy.log(Cl_o0/Cl_i0)) 

        def I_pump_form(Na_i,K_o):
            return rho*(1.0/(1.0+numpy.exp((Cnap - Na_i) / DCnap))*(1.0/(1.0+numpy.exp((Ckp - K_o)/DCkp)))) 

        def V_dot_form(I_Na,I_K,I_Cl,I_pump):
            return (-1.0/Cm)*(I_Na+I_K+I_Cl+I_pump) 

        beta= w_i / w_o 
        DNa_i = -DKi 
        DNa_o = -beta * DNa_i
        DK_o = -beta * DKi
        K_i = K_i0 + DKi 
        Na_i = Na_i0 + DNa_i 
        Na_o = Na_o0 + DNa_o 
        K_o = K_o0 + DK_o + Kg 

        ninf=n_inf(V)
        I_K = I_K_form(V,n,K_o,K_i)
        I_Na = I_Na_form(V,Na_o,Na_i,n)
        I_Cl = I_Cl_form(V)
        I_pump = I_pump_form(Na_i,K_o)
    
        r = R_minus*x/numpy.pi
        Vdot = (-1.0/Cm)*(I_Na+I_K+I_Cl+I_pump) 

        derivative = numpy.empty_like(state_variables)

        if_xdot = Delta+2*R_minus*(V-c_minus)*x-J*r*x 
        else_xdot = Delta+2*R_plus*(V-c_plus)*x-J*r*x
        derivative[0] = numpy.where(V <= Vstar, if_xdot, else_xdot)

        if_Vdot = Vdot - R_minus*x**2 + eta + J*r*(E-V) + (R_minus/numpy.pi)*Coupling_Term*(E-V)
        else_Vdot = Vdot - R_plus*x**2 + eta + J*r*(E-V) + (R_minus/numpy.pi)*Coupling_Term*(E-V)
        derivative[1] = numpy.where(V <= Vstar, if_Vdot, else_Vdot)

        derivative[2] = (ninf - n) / tau_n
        derivative[3] = -(gamma / w_i) * (I_K - 2.0 * I_pump)
        derivative[4] = epsilon * (K_bath - K_o)
        
        return derivative


def nextpow2(i):
    return int(2**np.ceil(np.log2(i)))

kionex2_param_names = [
    'E', 'K_bath', 'J', 'eta', 'Delta', 'c_minus', 'R_minus', 
    'c_plus', 'R_plus', 'Vstar', 'Cm', 'tau_n', 'gamma', 'epsilon'
]

def tvbk_run_sim_kionex2(sim, init_sim_state):
    conn = sim.connectivity
    s_w = scipy.sparse.csr_matrix(conn.weights)
    nr = conn.weights.shape[0]
    
    # TVB ensures conn.horizon >= 1 if coupling is used.
    # nextpow2 requires positive input.
    horizon_val = conn.horizon if conn.horizon > 0 else 1
    # If conn.horizon is 0 (e.g. no delays), TVB sets it to 1 for internal buffers.
    # We use nextpow2 for C++ buffer dimension which must be power of 2.
    # nextpow2(1) is 1.
    k_cx_horizon = nextpow2(horizon_val)

    k_cx = tvbk.Cx8s(nr, k_cx_horizon, 1) # batch_size = 1
    k_conn = tvbk.Conn(nr, s_w.data.size)
    k_conn.weights[:] = s_w.data.astype(np.float32)
    k_conn.indptr[:] = s_w.indptr.astype(np.uint32)
    k_conn.indices[:] = s_w.indices.astype(np.uint32)
    
    # Calculate integer delays for C++ kernel
    # If speed is inf or tract_lengths are 0, delays will be 0.
    non_zero_weights_mask = conn.weights != 0
    if np.any(non_zero_weights_mask):
        tracts_for_non_zero_weights = conn.tract_lengths[non_zero_weights_mask]
        # Ensure speed is not zero to avoid division by zero if tracts are non-zero
        # If speed is Inf, delays are 0. If speed is finite, calculate delay.
        if conn.speed == 0 and np.any(tracts_for_non_zero_weights > 0):
            raise ValueError("Connectivity speed is 0 with non-zero tract lengths.")
        
        if np.isinf(conn.speed):
            idelays_values = np.zeros_like(tracts_for_non_zero_weights, dtype=np.uint32)
        else:
            idelays_values = (
                tracts_for_non_zero_weights / conn.speed / sim.integrator.dt
            ).astype(np.uint32)
        k_conn.idelays[:] = idelays_values
    else: # No connections, idelays array will be empty or not used if s_w.data.size is 0
        pass

    # Initialize k_cx.buf (history buffer for C++ kernel)
    # sim.history.buffer shape: (tvb_horizon, nvar, N, nmode)
    # k_cx.buf shape: (batch, N, k_cx_horizon, width)
    # We use the history of the first state variable for coupling.
    k_cx.buf[:] = 0.0
    if conn.horizon > 0 and hasattr(sim, 'history') and sim.history is not None:
        # history_for_coupling shape: (N, tvb_horizon)
        history_for_coupling = sim.history.buffer[:, 0, :, 0].T 
        # Ensure we don't try to read more history than available or write past k_cx buffer
        len_to_copy = min(conn.horizon, k_cx_horizon)
        # Copy relevant part of history: (N, len_to_copy)
        # Assign to k_cx.buf[batch_idx, all_nodes, last_len_to_copy_timesteps, :]
        # Broadcasting the last dimension (1) to SIMD width (8)
        k_cx.buf[0, :, -len_to_copy:] = history_for_coupling[:, -len_to_copy:, np.newaxis]

    num_svar, num_parm = 5, 14 # For KIonEx2
    
    # x shape: (batch, num_svar, nr, simd_width)
    x = np.zeros((1, num_svar, nr, 8), 'f')
    # init_sim_state shape from TVB: (num_svar, nr, 1)
    x[0, :, :, 0] = init_sim_state[:, :, 0] 

    # p shape: (batch, nr, num_parm, simd_width)
    p = np.zeros((1, nr, num_parm, 8), 'f')
    for i, pname in enumerate(kionex2_param_names):
        param_val = getattr(sim.model, pname)
        # getattr might return scalar or (nr,) array. Broadcast to (nr, 8).
        p[0, :, i, :] = param_val.reshape(-1, 1) if hasattr(param_val, 'shape') else param_val

    num_time_steps_sim = int(sim.simulation_length / sim.integrator.dt)
    # For Raw monitor, period is dt, so num_skip is 1.
    num_skip = int(sim.monitors[0].period / sim.integrator.dt) if sim.monitors else 1
    
    output_time_points = num_time_steps_sim // num_skip
    y_cpp = np.zeros((output_time_points, num_svar, nr), 'f')
    
    # k_y stores result of one chunk of integration steps
    k_y = np.zeros_like(x) 
    # z is for noise terms if any (here nsig=0)
    z = np.zeros((1, num_svar, 8), 'f') 
    # seed for C++ RNG
    seed = np.zeros((1, 8, 4), np.uint64) 

    for t_idx in range(output_time_points):
        # tvbk.step_kionex2_8 is assumed to be the C++ kernel function
        tvbk.step_kionex2(k_cx, k_conn, x, k_y, z, p,
                            t_idx * num_skip, num_skip, sim.integrator.dt,
                            seed)
        y_cpp[t_idx] = k_y[0, :, :, 0] # Store first SIMD lane result

    return y_cpp


def test_kionex2_dfun():
    model_py = KIonEx2()
    num_svar = model_py._nvar
    num_parm = 14 # Based on kionex2.hpp
    width = 8 # dfun_kionex2_8 implies width 8

    for _ in range(128): # Run a few randomized tests
        dx_cpp = np.zeros((num_svar, width), 'f')
        # Initialize state variables with plausible random values
        # x: firing rate related, positive
        # V: membrane potential, around -70 to -20 mV
        # n: gating variable, 0 to 1
        # DKi: change in intracellular K, small negative or positive
        # Kg: extracellular K buffering, small negative or positive
        x_val = np.random.uniform(0.01, 2.0, size=(1,width)).astype('f')
        V_val = np.random.uniform(-70.0, -20.0, size=(1,width)).astype('f')
        n_val = np.random.uniform(0.0, 1.0, size=(1,width)).astype('f')
        DKi_val = np.random.uniform(-5.0, 5.0, size=(1,width)).astype('f')
        Kg_val = np.random.uniform(-5.0, 5.0, size=(1,width)).astype('f')
        
        x_cpp = np.vstack([x_val, V_val, n_val, DKi_val, Kg_val]).astype('f')
        
        c_cpp = np.random.randn(1, width).astype('f') / 2.0 # Coupling term

        # Parameters in same order as kionex2.hpp:
        # E,K_bath,J,eta,Delta,c_minus,R_minus,c_plus,R_plus,Vstar,Cm,tau_n,gamma,epsilon
        p_cpp = np.array([
            model_py.E[0],          # E
            model_py.K_bath[0],     # K_bath
            model_py.J[0],          # J
            model_py.eta[0],        # eta
            model_py.Delta[0],      # Delta
            model_py.c_minus[0],    # c_minus
            model_py.R_minus[0],    # R_minus
            model_py.c_plus[0],     # c_plus
            model_py.R_plus[0],     # R_plus
            model_py.Vstar[0],      # Vstar
            model_py.Cm[0],         # Cm
            model_py.tau_n[0],      # tau_n
            model_py.gamma[0],      # gamma
            model_py.epsilon[0]     # epsilon
        ], dtype='f').reshape(num_parm, 1).repeat(width, axis=1)
        
        # Randomize parameters slightly around defaults for robustness
        p_cpp += np.random.normal(scale=0.05, size=p_cpp.shape).astype('f')
        # Ensure certain parameters remain positive or within specific ranges if necessary
        p_cpp[1,:] = np.maximum(0.1, p_cpp[1,:]) # K_bath > 0
        p_cpp[2,:] = np.maximum(0.001, p_cpp[2,:]) # J > 0
        p_cpp[4,:] = np.maximum(0.0, p_cpp[4,:]) # Delta >=0
        p_cpp[6,:] = np.maximum(0.0001, p_cpp[6,:]) # R_minus > 0
        p_cpp[8,:] = np.minimum(-0.0001, p_cpp[8,:]) # R_plus < 0
        p_cpp[10,:] = np.maximum(0.1, p_cpp[10,:]) # Cm > 0
        p_cpp[11,:] = np.maximum(0.1, p_cpp[11,:]) # tau_n > 0
        p_cpp[12,:] = np.maximum(0.001, p_cpp[12,:]) # gamma > 0
        p_cpp[13,:] = np.maximum(0.0001, p_cpp[13,:]) # epsilon > 0


        m.dfun_kionex2_8(dx_cpp, x_cpp, c_cpp, p_cpp)

        # For Python model, need to set parameters on the instance
        # and reshape inputs appropriately
        model_py_instance = KIonEx2()
        model_py_instance.E = p_cpp[0,:]
        model_py_instance.K_bath = p_cpp[1,:]
        model_py_instance.J = p_cpp[2,:]
        model_py_instance.eta = p_cpp[3,:]
        model_py_instance.Delta = p_cpp[4,:]
        model_py_instance.c_minus = p_cpp[5,:]
        model_py_instance.R_minus = p_cpp[6,:]
        model_py_instance.c_plus = p_cpp[7,:]
        model_py_instance.R_plus = p_cpp[8,:]
        model_py_instance.Vstar = p_cpp[9,:]
        model_py_instance.Cm = p_cpp[10,:]
        model_py_instance.tau_n = p_cpp[11,:]
        model_py_instance.gamma = p_cpp[12,:]
        model_py_instance.epsilon = p_cpp[13,:]

        # Python dfun expects state_vars [nvar, nodes] and coupling [cvar, nodes]
        # and parameters to be attributes of the model object.
        # The C++ dfun takes parameters as an array [nparam, nodes].
        dx_py = model_py_instance.dfun(x_cpp, c_cpp)

        # Check that non-finite values (NaN, Inf) occur in the same places
        finite_cpp = np.isfinite(dx_cpp)
        finite_py = np.isfinite(dx_py)
        np.testing.assert_array_equal(finite_cpp, finite_py,
                                      err_msg="Mismatch in locations of non-finite values.")

        # Compare only the finite values
        np.testing.assert_allclose(dx_cpp[finite_cpp], dx_py[finite_py], rtol=1e-4, atol=1e-4)


def test_sim_tvb():
    import os, pandas as pd, tvb.simulator.lab as tvb
    path_data = "zz-sourin-kionex/HCP100Schaeffer/"
    subj = '0001'
    labels_path = path_data + 'Schaefer2018_100Parcels_17Networks_order.txt'
    df = pd.read_csv(labels_path, delimiter='\t', header=None)
    labels = df.iloc[:, 1]
    labels = labels.to_numpy(dtype='<U128')
    weights     = np.loadtxt(os.path.join(path_data, subj +'_1_dwi_connectome_100P_7NW_sift2_nolog.txt'), delimiter=' ')
    weights_log = np.loadtxt(os.path.join(path_data, subj +'_1_dwi_connectome_100P_7NW_sift2_log10.txt'), delimiter=' ')
    N=weights.shape[0]
    weights = weights/np.max(weights)
    print(weights.max(), weights.mean())
    conn = tvb.connectivity.Connectivity(
        weights = weights, 
        tract_lengths = np.zeros(np.shape(weights)),                              
        region_labels = labels, #np.array(np.zeros(np.shape(w)[0]),dtype='<U128'),
        centres = np.zeros(np.shape(weights)[0]),
        speed=np.array(np.Inf)
    )
    conn.compute_region_labels()
    conn.configure()
    K_bath = np.ones(N) * 5.5
    epi_reg = 62
    K_bath[epi_reg] = 15.5
    prop_reg = 85
    K_bath[prop_reg] = 12.5
    THH = KIonEx2(K_bath = K_bath)
    x_0=np.random.uniform(low=0.1, high=0.1, size=((1,1,N,1)))
    V_0=np.random.uniform(low=-15., high=-15., size=((1,1,N,1)))
    n_0=np.random.uniform(low=0.45, high=0.45, size=((1,1,N,1)))
    DKi_0=np.random.uniform(low=0.2, high=0.2, size=((1,1,N,1)))
    Kg_0=np.random.uniform(low=-1., high=-1., size=((1,1,N,1)))
    init_cond=np.concatenate([x_0, V_0, n_0, DKi_0, Kg_0], axis=1)
    dt      = 0.01
    nsigma  = 0.0
    G       = 2  #10 almost everyone sync; 4 almost no sync
    T = 0.5 # period for the sampling in * 10 ms
    Tinit = round(1000/T) # corresponds to 1 second of simulation 
    sim_len = 1 #  (5) * 1e3  
    conn.weights[epi_reg, prop_reg] = 0
    conn.weights[prop_reg, epi_reg] = 0
    sim = tvb.simulator.Simulator(
        connectivity = conn,
        model = THH,
        coupling = tvb.coupling.Scaling(a=np.r_[G]),
        integrator = tvb.integrators.HeunStochastic(
            dt = dt,
            noise = tvb.noise.Additive(nsig=np.r_[nsigma, nsigma, nsigma, 0, 0], noise_seed=42)
        ),
        monitors = [tvb.monitors.Raw()], # 0.1 was the default, monitors.TemporalAverage(period=1.0)
        simulation_length = sim_len,
        initial_conditions = init_cond,
        #stimulus = stimulus,
    ).configure()

    # Capture initial state after configuration
    init_sim_state_cpp = sim.current_state.copy() # Shape: (nvar, N, nmode=1)

    # Run TVB simulation (Python backend)
    (t_tvb, y_tvb_raw), = sim.run() 
    # y_tvb_raw shape: (time_points, nvar, N, nmode=1)
    # Reshape for comparison: (time_points, nvar, N)
    y_tvb = y_tvb_raw[:, :, :, 0]

    # Run C++ simulation with tvbk
    y_cpp = tvbk_run_sim_kionex2(sim, init_sim_state_cpp)

    # Compare results
    # Ensure same number of time points are compared
    num_compare_points = min(y_tvb.shape[0], y_cpp.shape[0])
    assert num_compare_points > 0, "No simulation data to compare."
    
    # Using tolerances from test_kionex2_dfun as a starting point
    rtol = 0.001
    atol = 0.02

    # Compare each time step
    for t_idx in range(num_compare_points):
        print(t_idx)
        np.testing.assert_allclose(
            y_tvb[t_idx], 
            y_cpp[t_idx], 
            rtol=rtol*(1+t_idx), 
            atol=atol*(1+t_idx),
            err_msg=f"Mismatch at time step {t_idx}"
        )
