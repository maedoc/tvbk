import pytest
import numpy as np
import tvbk as m
import tvb.simulator.models.oscillator as osc
import tvb.simulator.models.wilson_cowan as wc
import tvb.simulator.models.wong_wang as ww
import tvb.simulator.models.jansen_rit as jr_mod
import tvb.simulator.models.zerlaut as zer
import tvb.simulator.models.wong_wang_exc_inh as ww_exc_inh

def test_gast_schmidt_knosche_sd():
    model = it.GastSchmidtKnosche_SD()
    # Params: tau, tau_A, alpha, I, Delta, J, eta, cr, cv
    params = ['tau', 'tau_A', 'alpha', 'I', 'Delta', 'J', 'eta', 'cr', 'cv']
    check_model(model, 'dfun_gast_schmidt_knosche_sd_8', params)

def test_gast_schmidt_knosche_sf():
    model = it.GastSchmidtKnosche_SF()
    # Params: tau, tau_A, alpha, I, Delta, J, eta, cr, cv
    params = ['tau', 'tau_A', 'alpha', 'I', 'Delta', 'J', 'eta', 'cr', 'cv']
    check_model(model, 'dfun_gast_schmidt_knosche_sf_8', params)

def test_zetterberg_jansen():
    model = jr_mod.ZetterbergJansen()
    model.configure()
    # Params: He, Hi, ke, ki, e0, rho_1, rho_2, gamma_1..5, gamma_1T..3T, P, Q, U
    params = [
        'He', 'Hi', 'ke', 'ki', 'e0', 'rho_1', 'rho_2',
        'gamma_1', 'gamma_2', 'gamma_3', 'gamma_4', 'gamma_5',
        'gamma_1T', 'gamma_2T', 'gamma_3T',
        'P', 'Q', 'U'
    ]
    check_model(model, 'dfun_zetterberg_jansen_8', params)

def test_zerlaut_adaptation_first_order():
    model = zer.ZerlautAdaptationFirstOrder()
    # Params: g_L ... T, P_e, P_i
    params = [
        'g_L', 'E_L_e', 'E_L_i', 'C_m', 'b_e', 'a_e', 'b_i', 'a_i', 'tau_w_e', 'tau_w_i',
        'E_e', 'E_i', 'Q_e', 'Q_i', 'tau_e', 'tau_i',
        'N_tot', 'p_connect_e', 'p_connect_i', 'g', 'K_ext_e', 'K_ext_i',
        'external_input_ex_ex', 'external_input_ex_in', 'external_input_in_ex', 'external_input_in_in',
        'tau_OU', 'weight_noise', 'S_i', 'T',
        'P_e', 'P_i'
    ]
    check_model(model, 'dfun_zerlaut_adaptation_first_order_8', params)

def test_zerlaut_adaptation_second_order():
    model = zer.ZerlautAdaptationSecondOrder()
    # Params same as first order
    params = [
        'g_L', 'E_L_e', 'E_L_i', 'C_m', 'b_e', 'a_e', 'b_i', 'a_i', 'tau_w_e', 'tau_w_i',
        'E_e', 'E_i', 'Q_e', 'Q_i', 'tau_e', 'tau_i',
        'N_tot', 'p_connect_e', 'p_connect_i', 'g', 'K_ext_e', 'K_ext_i',
        'external_input_ex_ex', 'external_input_ex_in', 'external_input_in_ex', 'external_input_in_in',
        'tau_OU', 'weight_noise', 'S_i', 'T',
        'P_e', 'P_i'
    ]
    # Numerical differentiation might require looser tolerance
    check_model(model, 'dfun_zerlaut_adaptation_second_order_8', params, tolerance=1e-3)

@pytest.mark.parametrize("datatype", ["float32"])
def test_reduced_wong_wang_exc_inh(datatype):
    from tvb.simulator.models.wong_wang_exc_inh import ReducedWongWangExcInh
    model = ReducedWongWangExcInh()
    # Params: a_e, b_e, d_e, gamma_e, tau_e, w_p, W_e, J_N, I_o, G, I_ext,
    #         a_i, b_i, d_i, gamma_i, tau_i, W_i, J_i, lamda
    params = [
        'a_e', 'b_e', 'd_e', 'gamma_e', 'tau_e', 'w_p', 'W_e', 'J_N', 'I_o', 'G', 'I_ext',
        'a_i', 'b_i', 'd_i', 'gamma_i', 'tau_i', 'W_i', 'J_i', 'lamda'
    ]
    check_model(model, 'dfun_reduced_wong_wang_exc_inh_8', params, tolerance=1e-4)

@pytest.mark.parametrize("datatype", ["float32"])
def test_deco_balanced_exc_inh(datatype):
    from tvb.simulator.models.wong_wang_exc_inh import DecoBalancedExcInh
    model = DecoBalancedExcInh()
    # Params: ... + M_i
    params = [
        'a_e', 'b_e', 'd_e', 'gamma_e', 'tau_e', 'w_p', 'W_e', 'J_N', 'I_o', 'G', 'I_ext',
        'a_i', 'b_i', 'd_i', 'gamma_i', 'tau_i', 'W_i', 'J_i', 'lamda', 'M_i'
    ]
    check_model(model, 'dfun_deco_balanced_exc_inh_8', params, tolerance=1e-4)

@pytest.mark.parametrize("datatype", ["float32"])
def test_reduced_set_fitz_hugh_nagumo(datatype):
    from tvb.simulator.models.stefanescu_jirsa import ReducedSetFitzHughNagumo
    model = ReducedSetFitzHughNagumo() 
    model.configure() 
    # C++ Order: tau,a,b,K11,K12,K21,sigma,mu, e_i,f_i,m_i,n_i,IE_i,II_i, Aik,Bik,Cik
    params = [
        'tau','a','b','K11','K12','K21','sigma','mu',
        'e_i','f_i','m_i','n_i','IE_i','II_i',
        'Aik','Bik','Cik'
    ]
    # Arrays will be flattened by check_model parameters expansion logic
    check_model(model, 'dfun_reduced_set_fitz_hugh_nagumo_8', params, tolerance=1e-4)

@pytest.mark.parametrize("datatype", ["float32"])
def test_reduced_set_hindmarsh_rose(datatype):
    from tvb.simulator.models.stefanescu_jirsa import ReducedSetHindmarshRose
    model = ReducedSetHindmarshRose()
    model.configure()
    # C++ Order: r,s,K11,K12,K21, 
    # Vectors: a_i,b_i,c_i,d_i,e_i,f_i,h_i,p_i,m_i,n_i,IE_i,II_i
    # Matrices: A_ik, B_ik, C_ik (Note underscores compared to FHN)
    params = [
        'r','s','K11','K12','K21',
        'a_i','b_i','c_i','d_i','e_i','f_i','h_i','p_i','m_i','n_i','IE_i','II_i',
        'A_ik','B_ik','C_ik'
    ]
    check_model(model, 'dfun_reduced_set_hindmarsh_rose_8', params, tolerance=1e-4)

@pytest.mark.parametrize("datatype", ["float32"])
def test_dumont_gutkin(datatype):
    from tvb.simulator.models.infinite_theta import DumontGutkin
    model = DumontGutkin()
    # Params: I_e,Delta_e,eta_e,tau_e,I_i,Delta_i,eta_i,tau_i,tau_s,J_ee,J_ei,J_ie,J_ii,Gamma
    params = [
        "I_e","Delta_e","eta_e","tau_e","I_i","Delta_i","eta_i","tau_i",
        "tau_s","J_ee","J_ei","J_ie","J_ii","Gamma"
    ]
    check_model(model, 'dfun_dumont_gutkin_8', params, tolerance=1e-4)

def check_model(tvb_model, tvbk_dfun_name, param_order, state_order=None, n_node=8, tolerance=1e-5):
    # Setup inputs
    width = 8
    
    # Handle modes
    modes = getattr(tvb_model, 'number_of_modes', 1)
    # If modes > 1, we treat C++ kernel as handling all modes for 'width' nodes.
    # Python model needs 'width * modes' nodes to simulate the same data.
    py_n_node = width * modes
    
    # Random state
    # Python state: (n_sv_base, py_n_node)
    n_sv_base = len(tvb_model.state_variables)
    
    if hasattr(tvb_model, 'state_variable_range'):
        x_py = np.zeros((n_sv_base, py_n_node), dtype='f')
        for i, var_name in enumerate(tvb_model.state_variables):
            rng = tvb_model.state_variable_range.get(var_name)
            if rng is not None:
                low, high = rng
                # Basic check for reasonable bounds
                if not np.isfinite(low): low = -1.0
                if not np.isfinite(high): high = 1.0
                x_py[i, :] = np.random.uniform(low, high, size=py_n_node)
            else:
                x_py[i, :] = np.random.randn(py_n_node)
    else:
        x_py = np.random.randn(n_sv_base, py_n_node).astype('f')
    
    # Random coupling
    if hasattr(tvb_model, 'cvar'):
        n_cv = len(tvb_model.cvar)
    else:
        n_cv = 1 # default
    c_py = np.random.randn(n_cv, py_n_node).astype('f')
    
    # Parameters
    p_list = []
    for par_name in param_order:
        val = getattr(tvb_model, par_name)
        val = np.array(val)

        if val.ndim == 2:
            # Matrix parameter (e.g. Aik 3x3). Flatten to scalars.
            # Assuming these are constant across nodes (global params)
            # Flatten row-major
            val_flat = val.ravel()
            for v in val_flat:
                p_list.append(np.full(width, v, dtype='f'))
        elif val.ndim == 1 and val.size > 1 and val.size != width:
            # Vector parameter (e.g. E size 3) BUT not per-node (width=8)
            # Expand elements
            for v in val:
                p_list.append(np.full(width, v, dtype='f'))
        else:
            # Scalar or per-node parameter
            val = val.ravel()
            if len(val) == 1:
                val = np.repeat(val, width)
            elif len(val) != width:
                val = np.resize(val, width) # Just in case
            p_list.append(val)

    p = np.array(p_list).astype('f')

    # Prepare C++ input x and c
    # x_py shape (n_sv_base, width*modes)
    # Reshape to (n_sv_base, modes, width)
    # Transpose to (modes, n_sv_base, width) -> this groups mode0 vars, then mode1 vars
    # Reshape to (modes*n_sv_base, width)
    
    x_reshaped = x_py.reshape(n_sv_base, modes, width)
    x_cpp = x_reshaped.transpose(1, 0, 2).reshape(modes*n_sv_base, width).astype('f')
    
    # Coupling c. Assume coupling is per physical node? 
    # For ReducedSet, c_0 is summed coupling. 
    # If Python model has c_py shape (n_cv, width*modes). 
    # C++ kernel expects c shape (n_cv, width). 
    # Usually coupling is node-based. If modes are internal, they share coupling or have specific structure.
    # We'll validly assume c input to C++ is just the first 'width' slice if it's homogeneous?
    # Or does C++ kernel need to see coupling for each mode? 
    # In ReducedSet, C++ dfun takes `c[i]`. Single scalar coupling.
    # Python dfun `c_0` from `coupling`. 
    # If we pass random `c`, we should ensure consistency.
    # Let's assume we take the first `width` values for C++ c. 
    # And for Python c_py, we replicate them?
    # Or we generate c_cpp and expand to c_py.
    
    c_cpp = np.random.randn(n_cv, width).astype('f')
    # Expand for python: (n_cv, modes*width)
    # Replicate c_cpp 'modes' times?
    # ReducedSet Python: `c_0 = coupling[0, ...]`
    # It adds `c_0` to all modes? `+ c_0` in derivative.
    # So yes, repeat c_cpp.
    c_py = np.repeat(c_cpp, modes, axis=1)

    # Debug info
    print(f"DEBUG: Model {tvb_model.__class__.__name__}")
    print(f"DEBUG: x_cpp shape={x_cpp.shape}")
    print(f"DEBUG: p shape={p.shape}")
    print(f"DEBUG: modes={modes}")


    # Prepare output buffer
    n_sv_tot = modes * n_sv_base
    dx_cpp = np.zeros((n_sv_tot, width), 'f')

    # Run TVBK
    if isinstance(tvbk_dfun_name, str):
        getattr(m, tvbk_dfun_name)(dx_cpp, x_cpp, c_cpp, p)
    else:
        tvbk_dfun_name(dx_cpp, x_cpp, c_cpp, p)
        
    # Run TVB Python
    if modes > 1:
        # Special handling for ReducedSet models: loop over physical nodes and rely on broadcasting.
        dx_py_list = []
        for i in range(width):
            # Extract state for one physical node: (n_sv_base, modes)
            x_node = x_reshaped[:, :, i]
            # Extract coupling for one physical node: (n_cv,)
            c_node = c_cpp[:, i]

            try:
                # Reshape state to (n_sv, 1, modes) and coupling to (n_cv, 1, 1) to broadcast coupling across modes.
                # This matches C++ logic where coupling c[i] is added to all modes for node i.
                x_in = x_node.reshape(n_sv_base, 1, modes)
                c_in = c_node.reshape(n_cv, 1, 1)
                
                res = tvb_model.dfun(x_in, c_in, local_coupling=0.0)
                res = res.reshape(n_sv_base, modes)
            except (ValueError, IndexError) as e:
                print(f"DEBUG: ReducedSet reshape attempt failed: {e}")
                c_in_2 = np.repeat(c_node[:, np.newaxis, np.newaxis], modes, axis=1)
                res = tvb_model.dfun(x_node, c_in_2, local_coupling=0.0)

            
            if res.ndim == 3:
                res = res.reshape(n_sv_base, modes)
            dx_py_list.append(res)

        dx_py_array = np.stack(dx_py_list, axis=-1) # Shape (n_sv_base, modes, width)
        dx_py_transformed = dx_py_array.transpose(1, 0, 2).reshape(n_sv_tot, width)

    else:
        # Standard models
        try:
            x_tvb = x_py.reshape(x_py.shape + (1,))
            c_tvb = c_py.reshape(c_py.shape + (1,))
            dx_py = tvb_model.dfun(x_tvb, c_tvb, local_coupling=0.0)
        except (ValueError, IndexError):
            dx_py = tvb_model.dfun(x_py, c_py, local_coupling=0.0)
        
        if dx_py.ndim == 3:
            dx_py = dx_py.reshape(dx_py.shape[:-1])
            
        dx_py = dx_py.reshape(n_sv_base, modes * width)
        dx_py_reshaped = dx_py.reshape(n_sv_base, modes, width)
        dx_py_transformed = dx_py_reshaped.transpose(1, 0, 2).reshape(n_sv_tot, width)

    
    np.allclose(dx_cpp, dx_py_transformed, atol=tolerance, rtol=tolerance)
    if not np.allclose(dx_cpp, dx_py_transformed, atol=tolerance, rtol=tolerance):
        diff = np.abs(dx_cpp - dx_py_transformed)
        max_diff = np.max(diff)
        print(f"Max diff: {max_diff}")
        print("Diff map:", np.where(diff > tolerance))
        assert False, f"Mismatch with tolerance {tolerance}, max diff {max_diff}"
    if not np.allclose(dx_cpp, dx_py_transformed, atol=tolerance, rtol=tolerance):
        diff = np.abs(dx_cpp - dx_py_transformed)
        max_diff = np.max(diff)
        print(f"Max diff: {max_diff}")
        print("Diff map:", np.where(diff > tolerance))
        assert False, f"Mismatch with tolerance {tolerance}, max diff {max_diff}"

def test_kuramoto():
    model = osc.Kuramoto()
    # Parameters: omega
    check_model(model, 'dfun_kuramoto8', ['omega'])

def test_sup_hopf():
    model = osc.SupHopf()
    # Parameters: a, omega
    check_model(model, 'dfun_sup_hopf8', ['a', 'omega'])

def test_generic_2d():
    model = osc.Generic2dOscillator()
    # Parameters: tau, I, a, b, c, d, e, f, g, alpha, beta, gamma
    # Note: 'c' in parameters might conflict with 'c' variable (coupling). 
    # TVB model trait is 'c'.
    check_model(model, 'dfun_generic_2d8', 
                ['tau', 'I', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'alpha', 'beta', 'gamma'])

def test_wilson_cowan():
    model = wc.WilsonCowan()
    # Parameters: c_ee, c_ei, c_ie, c_ii, tau_e, tau_i, a_e, b_e, c_e, theta_e, a_i, b_i, theta_i, c_i, r_e, r_i, k_e, k_i, P, Q, alpha_e, alpha_i, shift_sigmoid
    # shift_sigmoid is boolean but we pass as float (0.0 or 1.0)
    # Python code check_model handles it if it's implicitly castable, but trait is boolean.
    # getattr will return bool/numpy bool. np.array([True]).astype('f') is 1.0. Correct.
    
    params = ['c_ee', 'c_ei', 'c_ie', 'c_ii', 'tau_e', 'tau_i', 'a_e', 'b_e', 
              'c_e', 'theta_e', 'a_i', 'b_i', 'theta_i', 'c_i', 'r_e', 'r_i', 
              'k_e', 'k_i', 'P', 'Q', 'alpha_e', 'alpha_i', 'shift_sigmoid']
    
    check_model(model, 'dfun_wilson_cowan8', params)

def test_reduced_wong_wang():
    model = ww.ReducedWongWang()
    # Parameters: a, b, d, gamma, tau_s, w, J_N, I_o
    check_model(model, 'dfun_reduced_wong_wang8', 
                ['a', 'b', 'd', 'gamma', 'tau_s', 'w', 'J_N', 'I_o'])

import tvb.simulator.models.epileptor as epi

import tvb.simulator.models.epileptor_rs as epi_rs

def test_epileptor():
    model = epi.Epileptor()
    # Parameters: x0, Iext, Iext2, a, b, slope, tt, Kvf, c, d, r, Ks, Kf, aa, bb, tau, modification
    params = ['x0', 'Iext', 'Iext2', 'a', 'b', 'slope', 'tt', 'Kvf', 'c', 'd', 'r', 'Ks', 'Kf', 'aa', 'bb', 'tau', 'modification']
    check_model(model, 'dfun_epileptor8', params)

def test_epileptor_2d():
    model = epi.Epileptor2D()
    # Parameters: x0, Iext, a, b, slope, c, d, r, Kvf, Ks, tt, modification
    params = ['x0', 'Iext', 'a', 'b', 'slope', 'c', 'd', 'r', 'Kvf', 'Ks', 'tt', 'modification']
    check_model(model, 'dfun_epileptor_2d8', params)

def test_epileptor_rs():
    model = epi_rs.EpileptorRestingState()
    params = [
        'x0', 'Iext', 'Iext2', 'a', 'b', 'slope', 'tt', 'Kvf', 'c', 'd', 'r', 'Ks', 'Kf', 'aa', 'bb', 'tau',
        'tau_rs', 'I_rs', 'a_rs', 'b_rs', 'd_rs', 'e_rs', 'f_rs', 'beta_rs', 'alpha_rs', 'gamma_rs', 'K_rs'
    ]
    check_model(model, 'dfun_epileptor_rs8', params)

import tvb.simulator.models.epileptorcodim3 as epi_c3

def test_epileptor_codim3():
    model = epi_c3.EpileptorCodim3()
    model.configure()
    # Params: E(3), F(3), b, R, c, dstar, Ks, modification, N
    params = [
        'E', 'F', 'b', 'R', 'c', 'dstar', 'Ks', 'modification', 'N'
    ]
    check_model(model, 'dfun_epileptor_codim3_8', params)

def test_epileptor_codim3_slow_mod():
    model = epi_c3.EpileptorCodim3SlowMod()
    # It seems SlowMod might NOT calculate G, H, L, M in configure?
    # Let's try calling configure first.
    model.configure() 
    print(f"DEBUG: SlowMod G={model.G}")
    if model.G is None:
        # Manually call update if needed?
        model.update_derived_parameters()
        print(f"DEBUG: SlowMod G after manual update={model.G}")

    # Check if G (etc) are available. If not, this test will fail and I need to investigate how to get them.
    # From code analysis, G, H, L, M seem to be used in dfun.
    # The dfun implementation calls self.G[0] etc.
    # The class defines G = None initially.
    # So configure() MUST set them.
    
    # Params: G(3), H(3), L(3), M(3), b, R, c, cA, cB, dstar, Ks, modification, N
    params = [
        'G', 'H', 'L', 'M', 'b', 'R', 'c', 'cA', 'cB', 'dstar', 'Ks', 'modification', 'N'
    ]
    check_model(model, 'dfun_epileptor_codim3_slow_mod_8', params)

import tvb.simulator.models.linear as linear
import tvb.simulator.models.hopfield as hopfield
import tvb.simulator.models.larter_breakspear as lb
import tvb.simulator.models.infinite_theta as it

def test_coombes_byrne():
    model = it.CoombesByrne()
    # Params: Delta, alpha, v_syn, k, eta
    params = ['Delta', 'alpha', 'v_syn', 'k', 'eta']
    check_model(model, 'dfun_coombes_byrne_8', params)

def test_coombes_byrne_2d():
    model = it.CoombesByrne2D()
    # Params: Delta, v_syn, k, eta
    params = ['Delta', 'v_syn', 'k', 'eta']
    check_model(model, 'dfun_coombes_byrne_2d_8', params)

def test_linear():
    model = linear.Linear()
    check_model(model, 'dfun_linear8', ['gamma'])

def test_hopfield():
    model = hopfield.Hopfield()
    # dynamic=0 (default)
    # Params: taux, tauT, dynamic
    # Param order should be respected: taux, tauT, dynamic
    # TVB model traited attributes doesn't guarantee order if we iterate?
    # check_model uses param_order list.
    params = ['taux', 'tauT', 'dynamic']
    check_model(model, 'dfun_hopfield8', params)

def test_hopfield_dynamic():
    model = hopfield.Hopfield(dynamic=np.array([1]))
    model.configure() 
    # configure() switches dfun to dfunDyn and updates nvar/cvar
    params = ['taux', 'tauT', 'dynamic']
    check_model(model, 'dfun_hopfield_dynamic8', params)

def test_larter_breakspear():
    model = lb.LarterBreakspear()
    # 32 params in C++ order:
    params = [
        "gCa","gK","gL","phi","gNa","TK","TCa","TNa","VCa","VK","VL","VNa",
        "d_K","tau_K","d_Na","d_Ca","aei","aie","b","C","ane","ani","aee",
        "Iext","rNMDA","VT","d_V","ZT","d_Z","QV_max","QZ_max","t_scale"
    ]
    check_model(model, 'dfun_larter_breakspear8', params, tolerance=1e-4)

def test_infinite_theta():
    model = it.DumontGutkin()
    # 14 params in C++ order:
    params = [
       "I_e","Delta_e","eta_e","tau_e","I_i","Delta_i","eta_i","tau_i",
       "tau_s","J_ee","J_ei","J_ie","J_ii","Gamma"
    ]
    check_model(model, 'dfun_infinite_theta8', params, tolerance=1e-4)
