
import numpy as np
import tvbk
import time

def run_benchmark():
    # Parameters for a stable simulation
    n_node = 68
    n_time = 1024
    dt = 0.01
    nt = 1000 
    n_batch = 1
    width = 8
    n_svar = 6 # JR
    n_parm = 14 # JR
    
    np.random.seed(42)
    
    # 1. Setup Connectivity (No 0-delays)
    from scipy import sparse
    W = sparse.random(n_node, n_node, density=0.2, format='csr', dtype=np.float32)
    W.data[:] = 0.01 # Small weights
    D = np.random.randint(1, 20, size=W.data.shape).astype(np.uint32)
    
    conn = tvbk.Conn(n_node, W.nnz)
    conn.weights[:] = W.data
    conn.indices[:] = W.indices
    conn.indptr[:] = W.indptr
    conn.idelays[:] = D

    # 2. Initial State & Params
    x_init = (np.random.randn(n_batch, n_svar, n_node, width).astype(np.float32) * 0.01)
    
    base_params = [3.25, 22.0, 100.0, 50.0, 6.0, 5.0, 0.56, 135.0, 1.0, 0.8, 0.25, 0.25, 0.0, 0.0]
    p_mono = np.zeros((n_batch, 1, n_parm, width), dtype=np.float32)
    for i in range(n_parm):
        p_mono[:, :, i, :] = base_params[i]
    
    p_integ = p_mono[0, 0].reshape(1, n_parm, width)

    seed_val = 42
    niter=100
    
    # --- MONOLITHIC ---
    print("Running Monolithic...")
    cxs_mono = tvbk.Cx8s(n_node, n_time, n_batch)
    cxs_mono.buf[:] = 0.0
    x_mono = x_init.copy()
    y_mono = np.zeros_like(x_mono)
    z_mono = np.zeros((n_batch, n_svar, width), dtype=np.float32)
    seed_mono = np.zeros((n_batch, width, 4), dtype=np.uint64)
    seed_mono[:] = seed_val

    t0 = time.time()
    for _ in range(niter):
        tvbk.step_jr(cxs_mono, conn, x_mono, y_mono, z_mono, p_mono, 0, nt, dt, seed_mono)
    t_mono = (time.time() - t0)/niter
    print(f"Monolithic time: {t_mono:.4f}s")
    
    # --- SPLIT LOOP ---
    print("Running Python Loop...")
    cx_loop = tvbk.Cx8(n_node, n_time)
    cx_loop.buf[:] = 0.0
    x_loop = x_init[0].copy()
    z_loop = np.zeros((n_svar, width), dtype=np.float32)
    seed_loop = np.zeros((width, 4), dtype=np.uint64)
    seed_loop[:] = seed_val
    
    t0 = time.time()
    for _ in range(niter):
        for t in range(nt):
            tvbk.cx_j8(cx_loop, conn, t)
            tvbk.step_integrate_jr8(cx_loop, x_loop, z_loop, p_integ, t, dt, seed_loop)
    t_loop = (time.time() - t0)/niter
    print(f"Python Loop time: {t_loop:.4f}s")
    
    if t_loop > 0:
        print(f"Speedup (Mono/Loop): {t_mono/t_loop:.2f}x")
        
    # --- Comparison ---
    diff = np.abs(x_mono[0] - x_loop)
    max_diff = np.max(diff)
    print(f"Final state Max Difference: {max_diff}")
    
    if max_diff < 1e-3:
        print("SUCCESS: Results match!")
    else:
        print("FAILURE: Results differ.")

if __name__ == "__main__":
    run_benchmark()
