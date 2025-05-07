import pytest
pytestmark = pytest.mark.tvb

import numpy as np
import tvbk as m

# Copied from ../../Nextcloud/2025/zz-sourin-kionex/Biophysical_wholebrain.md
# with minor adjustments for standalone use (e.g. Model base class)

# A minimal Model base class to allow KIonEx2 to be defined
class Model:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class KIonEx2(Model):
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
    
    E = np.array([0.])
    K_bath = np.array([5.5])
    J = np.array([0.1])
    eta = np.array([0.0])
    Delta = np.array([1.0])
    c_minus = np.array([-40.0])
    R_minus = np.array([0.5])
    c_plus = np.array([-20.0])
    R_plus = np.array([-0.5])
    Vstar = np.array([-31])
    Cm = np.array([1])
    tau_n = np.array([4])
    gamma = np.array([0.04])
    epsilon = np.array([0.001])

    state_variables = ['x', 'V','n','DKi','Kg']
    _nvar = 5
    cvar = np.array([0], dtype=np.int32)
    
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        x = state_variables[0, :]
        V = state_variables[1, :]
        n = state_variables[2, :]
        DKi = state_variables[3, :]
        Kg = state_variables[4, :]

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
     
        Coupling_Term = coupling[0, :]

        Cnap = 21.0; DCnap = 2.0; Ckp = 5.5; DCkp = 1.0 
        Cmna = -24.0; DCmna = 12.0; Chn = 0.4; DChn = -8.0 
        Cnk = -19.0; DCnk = 18.0; g_Cl = 7.5; g_Na = 40.0
        g_K = 22.0; g_Nal = 0.02; g_Kl = 0.12; rho = 250.
        w_i = 2160.0; w_o = 720.0; Na_i0 = 16.0; Na_o0 = 138.0
        K_i0 = 130.0; K_o0 = 4.80; Cl_i0 = 5.0; Cl_o0 = 112.0
        
        def m_inf(V_val): return 1.0/(1.0+np.exp((Cmna-V_val)/DCmna))
        def n_inf(V_val): return 1.0/(1.0+np.exp((Cnk-V_val)/DCnk))
        def h_fun(n_val): return 1.1 - 1.0 / (1.0 + np.exp(-8.0 * (n_val - 0.4)))

        beta= w_i / w_o 
        DNa_i = -DKi 
        DNa_o = -beta * DNa_i
        DK_o = -beta * DKi
        K_i_val = K_i0 + DKi 
        Na_i_val = Na_i0 + DNa_i 
        Na_o_val = Na_o0 + DNa_o 
        K_o_val = K_o0 + DK_o + Kg 

        ninf_val=n_inf(V)
        I_K = (g_Kl+g_K*n)*(V- 26.64*np.log(K_o_val/K_i_val)) 
        I_Na = (g_Nal+g_Na*m_inf(V)*h_fun(n))*(V- 26.64*(np.log(Na_o_val) - np.log(Na_i_val)))
        I_Cl = g_Cl*(V+ 26.64*np.log(Cl_o0/Cl_i0)) 
        I_pump = rho*(1.0/(1.0+np.exp((Cnap - Na_i_val) / DCnap))*(1.0/(1.0+np.exp((Ckp - K_o_val)/DCkp)))) 
    
        r = R_minus*x/np.pi
        Vdot = (-1.0/Cm)*(I_Na+I_K+I_Cl+I_pump) 

        derivative = np.empty_like(state_variables)

        if_xdot = Delta+2*R_minus*(V-c_minus)*x-J*r*x 
        else_xdot = Delta+2*R_plus*(V-c_plus)*x-J*r*x
        derivative[0] = np.where(V <= Vstar, if_xdot, else_xdot)

        if_Vdot = Vdot - R_minus*x**2 + eta + J*r*(E-V) + (R_minus/np.pi)*Coupling_Term*(E-V)
        else_Vdot = Vdot - R_plus*x**2 + eta + J*r*(E-V) + (R_minus/np.pi)*Coupling_Term*(E-V)
        derivative[1] = np.where(V <= Vstar, if_Vdot, else_Vdot)

        derivative[2] = (ninf_val - n) / tau_n
        derivative[3] = -(gamma / w_i) * (I_K - 2.0 * I_pump)
        derivative[4] = epsilon * (K_bath - K_o_val)
        
        return derivative

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
