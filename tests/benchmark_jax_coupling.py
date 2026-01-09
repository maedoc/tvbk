import pytest
import jax
import jax.numpy as jp
import numpy as np
import scipy.sparse
from tvbk.jax import coupling

@pytest.fixture(scope="module")
def coupling_data():
    # Setup parameters roughly mimicking a realistic scenario
    num_node = 1000
    num_item = 8
    num_nonzero = num_node * 50  # ~50 connections per node
    horizon = 2048
    
    # Random connectivity
    # Use fixed seed for reproducibility across benchmarks
    np.random.seed(42)
    indices = np.random.randint(0, num_node, size=num_nonzero)
    indptr = np.concatenate(([0], np.sort(np.random.randint(0, num_nonzero, size=num_node-1)), [num_nonzero]))
    indptr = np.sort(indptr) # ensure sorted
    data = np.random.rand(num_nonzero).astype(np.float32)
    csr_weights = scipy.sparse.csr_matrix((data, indices, indptr), shape=(num_node, num_node))
    
    # Delays
    idelays = np.random.randint(1, 100, size=num_nonzero).astype(np.int32)
    
    # JAX arrays
    j_indices = jp.array(csr_weights.indices)
    j_weights = jp.array(csr_weights.data)
    j_indptr = jp.array(csr_weights.indptr)
    
    # Precompute row indices for scatter (Python impl)
    _csr_rows = np.concatenate([i*np.ones(n, dtype=np.int32)
                                for i, n in enumerate(np.diff(csr_weights.indptr))])
    j_csr_rows = jp.array(_csr_rows)
    
    horizonm1 = horizon - 1
    idelays2_py = jp.array(horizon + np.c_[idelays, idelays-1].T) # For Python impl
    
    # Casts for C++ impl
    j_indices_u32 = j_indices.astype(jp.uint32)
    j_indptr_u32 = j_indptr.astype(jp.uint32)
    idelays_u32 = jp.array(idelays, dtype=jp.uint32)
    
    # Buffer
    buffer_shape = (num_node, horizon, num_item)
    key = jax.random.PRNGKey(0)
    buffer = jax.random.normal(key, buffer_shape)
    
    t0 = 2000
    
    return {
        "buffer": buffer,
        "j_weights": j_weights,
        "j_indices": j_indices,
        "j_indptr": j_indptr,
        "j_csr_rows": j_csr_rows,
        "idelays2_py": idelays2_py,
        "horizonm1": horizonm1,
        "num_node": num_node,
        "num_item": num_item,
        "j_indices_u32": j_indices_u32,
        "j_indptr_u32": j_indptr_u32,
        "idelays_u32": idelays_u32,
        "t0": t0
    }

def test_coupling_python_jax(benchmark, coupling_data):
    d = coupling_data
    buffer = d["buffer"]
    j_weights = d["j_weights"]
    j_indices = d["j_indices"]
    j_csr_rows = d["j_csr_rows"]
    idelays2 = d["idelays2_py"]
    horizonm1 = d["horizonm1"]
    num_node = d["num_node"]
    num_item = d["num_item"]
    t0 = d["t0"]

    @jax.jit
    def cfun(buffer, t):
        t_indices = (t - idelays2) & horizonm1 
        val = buffer[j_indices, t_indices] 
        w = j_weights.reshape(1, -1, 1) 
        wxij = w * val 
        cx = jp.zeros((2, num_node, num_item))
        cx = cx.at[:, j_csr_rows].add(wxij)
        return cx

    # Warmup
    _ = cfun(buffer, t0).block_until_ready()
    
    # Simple wrapper to handle block_until_ready
    def run_step():
        cfun(buffer, t0).block_until_ready()

    benchmark(run_step)

def test_coupling_cpp_custom_call(benchmark, coupling_data):
    d = coupling_data
    buffer = d["buffer"]
    j_weights = d["j_weights"]
    j_indices = d["j_indices_u32"]
    j_indptr = d["j_indptr_u32"]
    idelays = d["idelays_u32"]
    t0 = d["t0"]

    @jax.jit
    def cfun_cpp(buffer, t):
        return coupling(
            buffer, 
            j_weights, 
            j_indices, 
            j_indptr, 
            idelays,
            jp.array(t, dtype=jp.uint32)
        )

    # Warmup
    _ = cfun_cpp(buffer, t0).block_until_ready()

    def run_step():
        cfun_cpp(buffer, t0).block_until_ready()

    benchmark(run_step)

def test_coupling_cpp_batched(benchmark, coupling_data):
    d = coupling_data
    buffer = d["buffer"]
    j_weights = d["j_weights"]
    j_indices = d["j_indices_u32"]
    j_indptr = d["j_indptr_u32"]
    idelays = d["idelays_u32"]
    t0 = d["t0"]
    
    # "batch_size" here refers to the number of SIMD vectors (width=8).
    # Total simulations = batch_size * 8.
    batch_size = 16
    batched_buffer = jp.tile(buffer[None, ...], (batch_size, 1, 1, 1))
    
    assert batched_buffer.shape == (batch_size, d["num_node"], 2048, 8)
    # Confirming logic: 16 batches of 8 items = 128 simulations

    @jax.jit
    def cfun_cpp_batch(b_buf, t):
        # We explicitly map the coupling function call
        return jax.vmap(lambda b: coupling(
            b, 
            j_weights, 
            j_indices, 
            j_indptr, 
            idelays,
            jp.array(t, dtype=jp.uint32)
        ))(b_buf)

    # Warmup
    _ = cfun_cpp_batch(batched_buffer, t0).block_until_ready()

    def run_step():
        cfun_cpp_batch(batched_buffer, t0).block_until_ready()
    
    # Benchmark rounds per step logic implicitly handled by pytest-benchmark,
    # but we can configure it if needed. For now default is fine.
    # Note: This benchmarks B simulations per step.
    benchmark(run_step)

def test_coupling_python_jax_batched(benchmark, coupling_data):
    d = coupling_data
    buffer = d["buffer"]
    j_weights = d["j_weights"]
    j_indices = d["j_indices"]
    j_csr_rows = d["j_csr_rows"]
    idelays2 = d["idelays2_py"]
    horizonm1 = d["horizonm1"]
    num_node = d["num_node"]
    num_item = d["num_item"]
    t0 = d["t0"]

    batch_size = 16
    batched_buffer = jp.tile(buffer[None, ...], (batch_size, 1, 1, 1))
    
    assert batched_buffer.shape == (batch_size, num_node, 2048, num_item)

    @jax.jit
    def cfun_batched(b_buf, t):
        # Python batched implementation (vmap cfun)
        def cfun_single(buf_slice):
            t_indices = (t - idelays2) & horizonm1
            val = buf_slice[j_indices, t_indices]
            w = j_weights.reshape(1, -1, 1)
            wxij = w * val
            cx = jp.zeros((2, num_node, num_item))
            cx = cx.at[:, j_csr_rows].add(wxij)
            return cx
        return jax.vmap(cfun_single)(b_buf)

    # Warmup
    _ = cfun_batched(batched_buffer, t0).block_until_ready()

    def run_step():
        cfun_batched(batched_buffer, t0).block_until_ready()

    benchmark(run_step)

def test_correctness(coupling_data):
    d = coupling_data
    buffer = d["buffer"]
    j_weights = d["j_weights"]
    j_indices = d["j_indices"]
    j_csr_rows = d["j_csr_rows"]
    idelays2 = d["idelays2_py"]
    horizonm1 = d["horizonm1"]
    num_node = d["num_node"]
    num_item = d["num_item"]
    t0 = d["t0"]
    
    # C++ inputs
    j_indices_u32 = d["j_indices_u32"]
    j_indptr_u32 = d["j_indptr_u32"]
    idelays = d["idelays_u32"]

    # Python Implementation
    @jax.jit
    def cfun_py(buffer, t):
        t_indices = (t - idelays2) & horizonm1 
        val = buffer[j_indices, t_indices] 
        w = j_weights.reshape(1, -1, 1) 
        wxij = w * val 
        cx = jp.zeros((2, num_node, num_item))
        cx = cx.at[:, j_csr_rows].add(wxij)
        return cx

    # C++ Implementation
    @jax.jit
    def cfun_cpp(buffer, t):
        return coupling(
            buffer, 
            j_weights, 
            j_indices_u32, 
            j_indptr_u32, 
            idelays,
            jp.array(t, dtype=jp.uint32)
        )

    res_py = cfun_py(buffer, t0)
    res_cpp = cfun_cpp(buffer, t0)
    
    np.testing.assert_allclose(res_cpp, res_py, atol=1e-4, err_msg="C++ output does not match Python output (Single)")
    
    # Batched Correctness
    batch_size = 4
    batched_buffer = jp.tile(buffer[None, ...], (batch_size, 1, 1, 1))
    
    @jax.jit
    def cfun_py_batch(b_buf, t):
        return jax.vmap(lambda b: cfun_py(b, t))(b_buf)
        
    @jax.jit
    def cfun_cpp_batch(b_buf, t):
        return jax.vmap(lambda b: coupling(
            b, j_weights, j_indices_u32, j_indptr_u32, idelays, jp.array(t, dtype=jp.uint32)
        ))(b_buf)

    res_py_batch = cfun_py_batch(batched_buffer, t0)
    res_cpp_batch = cfun_cpp_batch(batched_buffer, t0)

    np.testing.assert_allclose(res_cpp_batch, res_py_batch, atol=1e-4, err_msg="C++ output does not match Python output (Batched)")

    print("\nCorrectness test passed (Single and Batched)!")
