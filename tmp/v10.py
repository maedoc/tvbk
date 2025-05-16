import matplotlib as mpl
mpl.use('qt5agg', force=True)
import os
import ctypes
import numpy as np
import time
import v0
from k_ion_exchange import KIonEx # For accessing state variable ranges
import v8 # For comparison
import v8_ct # For comparison

lib_name = "_v10.so"
c_file_name = "v10.c"
h_file_name = "v10.h"
ct_file_name = "v10_ct.py"
compile_cmd_lib = f"gcc -shared -fPIC -o {lib_name} {c_file_name} /usr/lib/x86_64-linux-gnu/libOpenCL.so.1"
print(f"Compiling {c_file_name} into {lib_name}...")
assert os.system(compile_cmd_lib)==0
ctypesgen_cmd = f"ctypesgen {h_file_name} -l {lib_name} -l /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 -o {ct_file_name}" 
print(f"Generating ctypes bindings with ctypesgen: {ct_file_name}...")
assert os.system(ctypesgen_cmd)==0
import v10_ct

# --- Vector Addition Test (placeholder, can be removed or kept) ---
# This part is from previous steps and can be modified or removed as needed.
# For now, let's assume it's not the primary focus.
# print("\n--- Running Vector Addition Test ---")
# if hasattr(v10_ct, 'perform_vector_addition'):
#     v10_ct.lib.perform_vector_addition.argtypes = [
#         np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
#         np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
#         np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
#         ctypes.c_uint
#     ]
#     v10_ct.lib.perform_vector_addition.restype = ctypes.c_int
#     vector_size = 1024
#     h_a = np.arange(vector_size, dtype=np.float32)
#     h_b = np.arange(vector_size, dtype=np.float32) * 2.0
#     h_c_opencl = np.empty(vector_size, dtype=np.float32)
#     ret_val = v10_ct.lib.perform_vector_addition(h_a, h_b, h_c_opencl, vector_size)
#     if ret_val == 0:
#         print("OpenCL vector addition successful.")
#         h_c_expected = h_a + h_b
#         if np.allclose(h_c_opencl, h_c_expected):
#             print("Verification successful: Results match expected values.")
#         else:
#             print("Verification FAILED: Results do NOT match expected values.")
#     else:
#         print(f"OpenCL vector addition failed with error code: {ret_val}")
# else:
#     print("perform_vector_addition not found in v10_ct")


# --- sim_run Test ---
print("\n--- Running sim_run Test ---")

N = v0.sim.connectivity.weights.shape[0]
NSVAR = 5 
NTIME = int(v0.sim.simulation_length / v0.sim.integrator.dt)
MAXDELAY = v0.sim.connectivity.idelays.max()
H = MAXDELAY + 1
H2 = 2**int(np.ceil(np.log2(H))) 
NTAVG = NTIME // 100 
BATCH_SIZE = 32768 # Lowered batch size
NSVAR_TAVG = 1 # Number of state variables to store in tavg

CV_PARAM = v0.sim.coupling.a[0] 
DT_PARAM = v0.sim.integrator.dt
PROGRESS_PERIOD_PARAM = v0.sim.monitors[1].period 

weights_f = np.ascontiguousarray(v0.sim.connectivity.weights.T, dtype=np.float32).flatten()
idelays_u32 = np.ascontiguousarray(v0.sim.connectivity.idelays.T, dtype=np.uint32).flatten()

G_f = np.full(BATCH_SIZE, CV_PARAM, dtype=np.float32) 
K_bath_flat = v0.sim.model.K_bath.astype(np.float32) 
K_bath_f = np.tile(K_bath_flat[:, np.newaxis], (1, BATCH_SIZE)).flatten()

initial_states_v0 = v0.init_cond[0, :, :, 0].astype(np.float32) 
states_f = np.tile(initial_states_v0[:, :, np.newaxis], (1, 1, BATCH_SIZE))
states_f = np.ascontiguousarray(states_f.transpose(0, 2, 1).reshape(NSVAR * BATCH_SIZE, N)) # (nsvar*batch, N)
states_f = np.ascontiguousarray(states_f.reshape(NSVAR, BATCH_SIZE, N).transpose(0,2,1).flatten()) # (nsvar, N, batch_size) -> flat

# History buffer from v0.sim.history (N, horizon)
# We need (N, H2, BATCH_SIZE)
# v0.sim.history.buffer is (time, vars, nodes, regions) -> (horizon, 1, N, 1)
# Taking the initial history state.
history_v0_slice = v0.sim.history.buffer[:, 0, :, 0].T.astype(np.float32) # (N, horizon)
padded_history_v0 = np.zeros((N, H2), dtype=np.float32)
current_hist_len = history_v0_slice.shape[1]
paste_idx = min(H2, current_hist_len)
padded_history_v0[:, H2-paste_idx:] = history_v0_slice[:, current_hist_len-paste_idx:]

history_f = np.tile(padded_history_v0[:, :, np.newaxis], (1, 1, BATCH_SIZE)) # (N, H2, BATCH_SIZE)
history_f = np.ascontiguousarray(history_f.transpose(2,0,1).flatten()) # (batch_size, N, H2) -> flat for C order (nnode * h2 * batch_size)

# tavg will only store the first state variable
tavg_f = np.zeros((NTAVG, NSVAR_TAVG, N, BATCH_SIZE), dtype=np.float32)
# Flattening order for C: (ntavg, nsvar_tavg, nnode, batch_size)
# To match kernel's expected flat layout for tavg ( (tavg_idx * NSVAR_TAVG * N + node_idx) * BATCH_SIZE + batch_idx )
# if we consider NSVAR_TAVG in the kernel's indexing logic for tavg.
# Or, if kernel treats tavg as (tavg_idx, N, BATCH_SIZE) for the single svar:
# (tavg_idx * N * BATCH_SIZE) + (node_idx * BATCH_SIZE) + batch_idx
# Let's assume the C side will handle tavg as (NTAVG, N_NODES, BATCH_SIZE) for the single variable.
# Python side: (NTAVG, NSVAR_TAVG, N, BATCH_SIZE)
# C side expects flat array for (NTAVG * NSVAR_TAVG * N * BATCH_SIZE) elements.
# The host C code will calculate offset into this flat array.
# The kernel receives a segment of this, effectively (NSVAR_TAVG, N, BATCH_SIZE) or (N, BATCH_SIZE) if NSVAR_TAVG=1.
tavg_f_flat = np.ascontiguousarray(tavg_f.flatten()) # Simplest flattening, C side will manage offsets.

s_ct = v10_ct.sim_t()
s_ct.nnode = N
s_ct.nsvar = NSVAR
s_ct.ntime = NTIME
s_ct.maxdelay = MAXDELAY
s_ct.h2 = H2
s_ct.ntavg = NTAVG
s_ct.batch_size = BATCH_SIZE
s_ct.cv = CV_PARAM 
s_ct.dt = DT_PARAM
s_ct.progress_period = PROGRESS_PERIOD_PARAM

s_ct.weights = weights_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
s_ct.idelays = idelays_u32.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
s_ct.G = G_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
s_ct.K_bath = K_bath_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
s_ct.states = states_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
s_ct.history = history_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
s_ct.tavg = tavg_f_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# ctypesgen should have set up sim_run with its argtypes and restype.
# We access it directly from the v10_ct module.
if not hasattr(v10_ct, 'sim_run'):
    print("Error: sim_run function not found in v10_ct.py. Check v10.h declaration and ctypesgen output.")
    exit(1)

print(f"Calling sim_run with N={N}, batch_size={BATCH_SIZE}...")
tik = time.time()
# Call the function using the reference from the generated v10_ct module
ret_sim_run = v10_ct.sim_run(ctypes.byref(s_ct))
tok = time.time()

if ret_sim_run == 0:
    print("sim_run executed successfully.")
    duration = tok - tik
    iters_per_sec = (NTIME * BATCH_SIZE) / duration
    print(f"{iters_per_sec:.2f} iter/s")
    
    # Reshape tavg_f_flat back to its original conceptual shape for verification
    # Original C-side flat order was (ntavg, nsvar, nnode, batch_size) effectively,
    # based on kernel logic: (tavg_time_idx * s_nsvar * s_nnode * s_batch_size) + ... + batch_idx
    # Python tavg_f was (NTAVG, NSVAR, N, BATCH_SIZE), then tavg_f_flat = np.ascontiguousarray(tavg_f.transpose(0,1,3,2).flatten())
    # This transpose made it (NTAVG, NSVAR, BATCH_SIZE, N) then flatten.
    # So, to match kernel's view for verification, we need to be careful.
    # Kernel writes as if flat array is indexed by:
    # (tavg_time_idx, svar_idx, node_idx, batch_idx)
    # Let's reshape tavg_f_flat to (NTAVG, NSVAR_TAVG, N, BATCH_SIZE) for Python-side comparison.
    tavg_from_cl = tavg_f_flat.reshape(NTAVG, NSVAR_TAVG, N, BATCH_SIZE)

    # --- Verification ---
    print("\n--- Verifying tavg results against KIonEx state variable ranges (only for the first state variable) ---")
    all_ranges_ok = True
    # KIonEx.state_variables gives the order: ['x', 'V', 'n', 'DKi', 'Kg']
    # KIonEx.state_variable_range.default is a dict mapping name to [lo, hi]
    # We are only checking the first state variable ('x') as it's the only one in tavg_from_cl
    for svar_idx in range(NSVAR_TAVG): # This will loop only once for svar_idx = 0
        svar_name = KIonEx.state_variables[svar_idx] # This will be 'x'
        svar_limits = KIonEx.state_variable_range.default[svar_name]
        lower_bound = svar_limits[0]
        upper_bound = svar_limits[1]
        
        # tavg_from_cl has shape (NTAVG, NSVAR_TAVG, N, BATCH_SIZE)
        current_svar_data = tavg_from_cl[:, svar_idx, :, :] # svar_idx will be 0
        
        percentile_5 = np.percentile(current_svar_data, 5)
        percentile_95 = np.percentile(current_svar_data, 95)
        
        svar_percentiles_ok = True
        if percentile_5 < lower_bound:
            print(f"WARNING: State variable '{svar_name}' (idx {svar_idx}) 5th percentile ({percentile_5:.4f}) is BELOW lower bound ({lower_bound}).")
            svar_percentiles_ok = False
            all_ranges_ok = False
        if percentile_95 > upper_bound:
            print(f"WARNING: State variable '{svar_name}' (idx {svar_idx}) 95th percentile ({percentile_95:.4f}) is ABOVE upper bound ({upper_bound}).")
            svar_percentiles_ok = False
            all_ranges_ok = False
        
        if not svar_percentiles_ok:
            min_val = np.min(current_svar_data)
            max_val = np.max(current_svar_data)
            print(f"  Info: Actual min/max for '{svar_name}': [{min_val:.4f}, {max_val:.4f}]")

    if all_ranges_ok:
        print("All state variables' 5th-95th percentiles in tavg_from_cl are within their expected KIonEx ranges.")
    else:
        print("Some state variables in tavg_from_cl are out of their expected KIonEx ranges. See warnings above.")

    if np.all(tavg_from_cl == 0.0):
        print("ADDITIONAL WARNING: tavg_from_cl is all zeros. Kernel might not have executed as expected or all results are zero, irrespective of range checks.")

    # --- Comparison with v8.py (ISPC version) ---
    print("\n--- Comparing with v8.py (ISPC version) ---")
    # Ensure v8 compilation happens if needed (v8.py handles this on import/first use)
    # Setup Gs and Kbaths for v8.make_sim to match the first 8 batches of v10
    # v8 runs 8 instances internally.
    Gs_v8 = np.full((8,), CV_PARAM, dtype='f')
    Kbaths_v8 = np.tile(K_bath_flat[:, np.newaxis], (1, 8)).astype('f') # K_bath_flat is (N,)

    print("Running v8 (ISPC) simulation for comparison...")
    v8_data = v8.make_sim(Gs_v8, Kbaths_v8) # Uses v0.init_cond and v0.history internally
    
    tik_v8 = time.time()
    v8_ct.sim_run(v8_data['sp'])
    tok_v8 = time.time()
    duration_v8 = tok_v8 - tik_v8
    if duration_v8 > 0:
        iters_per_sec_v8 = (NTIME * 8) / duration_v8 # v8 runs 8 batches
        print(f"v8 (ISPC) run completed: {iters_per_sec_v8:.2f} iter/s (for 8 batches)")
    else:
        print("v8 (ISPC) run completed (duration too short for iter/s).")

    # Extract tavg from v8 for the first state variable
    # v8_data['tavg_f'] has shape (NTAVG, 5, N, 8)
    tavg_v8_svar0 = v8_data['tavg_f'][:, 0, :, :] # Shape: (NTAVG, N, 8)

    # Extract tavg from v10 (OpenCL) for the first state variable and first 8 batches
    # tavg_from_cl has shape (NTAVG, NSVAR_TAVG, N, BATCH_SIZE), NSVAR_TAVG=1
    tavg_v10_svar0_first8 = tavg_from_cl[:, 0, :, :8] # Shape: (NTAVG, N, 8)

    print("Comparing tavg results (first state variable, first 8 batches)...")
    try:
        np.testing.assert_allclose(tavg_v10_svar0_first8, tavg_v8_svar0, rtol=1e-5, atol=1e-5)
        print("SUCCESS: v10 (OpenCL) tavg results match v8 (ISPC) tavg results.")
    except AssertionError as e:
        print("FAILURE: v10 (OpenCL) tavg results DO NOT match v8 (ISPC) tavg results.")
        print(e)
        # For debugging, you might want to save the arrays:
        # np.save("tavg_v10_debug.npy", tavg_v10_svar0_first8)
        # np.save("tavg_v8_debug.npy", tavg_v8_svar0)
        # print("Saved tavg_v10_debug.npy and tavg_v8_debug.npy for inspection.")

    # --- Plotting comparison ---
    if ret_sim_run == 0: # Only plot if sim_run was successful
        print("\n--- Plotting comparison of v10 (OpenCL) and v8 (ISPC) tavg results ---")
        import matplotlib as mpl
        mpl.use('QtAgg', force=True) # Use Agg backend for non-interactive saving
        import pylab as pl

        pl.figure(figsize=(12, 10))
        
        # Time vector for tavg points
        # Each tavg point is an average over tpp = NTIME / NTAVG steps.
        # The time represents the midpoint or end of these segments.
        # For simplicity, let's use the end time of each segment.
        tpp = NTIME / NTAVG if NTAVG > 0 else NTIME 
        t_tavg = (np.arange(NTAVG) + 1) * tpp * DT_PARAM

        # Data to plot (first state variable, first batch instance)
        # tavg_v10_svar0_first8 has shape (NTAVG, N, 8)
        # tavg_v8_svar0 has shape (NTAVG, N, 8)
        data_v10_plot = tavg_v10_svar0_first8[:, :, 0] # First batch instance
        data_v8_plot = tavg_v8_svar0[:, :, 0]         # First batch instance

        vertical_offset_scale = np.std(data_v10_plot) * 2.0 # Heuristic for offset
        if vertical_offset_scale == 0: vertical_offset_scale = 1.0


        # Plot v10 (OpenCL) results - black
        for i in range(N):
            pl.plot(t_tavg, data_v10_plot[:, i] + i * vertical_offset_scale, 'k', alpha=0.7, linewidth=0.8)
        
        # Plot v8 (ISPC) results - red
        for i in range(N):
            pl.plot(t_tavg, data_v8_plot[:, i] + i * vertical_offset_scale, 'r', alpha=0.5, linewidth=0.8)

        pl.xlabel(f"Time (ms) - {NTAVG} tavg points")
        pl.ylabel(f"State Variable 0 (x) + Offset (per node, offset scale: {vertical_offset_scale:.2f})")
        pl.title("Comparison of Time-Averaged Output (SVar 0, Batch 0)\nBlack: v10 (OpenCL), Red: v8 (ISPC)")
        pl.grid(True, linestyle=':', alpha=0.5)
        pl.show()
        
        plot_filename = 'v10_vs_v8_comparison.jpg'
        pl.savefig(plot_filename)
        print(f"Comparison plot saved to {plot_filename}")
        # pl.show() # Typically not used with Agg backend or in scripts

else:
    print(f"sim_run failed with error code: {ret_sim_run}")


