
import numpy as np
import tvbk
import pytest

def test_buffer_initialization():
    """Verify that Cx/Cx8 buffers are zero-initialized."""
    n_node = 10
    n_time = 128
    
    # Test Cx8
    cx = tvbk.Cx8(n_node, n_time)
    # Check if buf is all zeros
    # Note: Accessing cx.buf returns a numpy view
    assert np.all(cx.buf == 0.0), "Cx8 buffer should be initialized to zero"
    assert np.all(cx.cx1 == 0.0), "Cx8 cx1 should be initialized to zero"
    assert np.all(cx.cx2 == 0.0), "Cx8 cx2 should be initialized to zero"
    
    # Test Cx8s
    n_batch = 2
    cxs = tvbk.Cx8s(n_node, n_time, n_batch)
    assert np.all(cxs.buf == 0.0), "Cx8s buffer should be initialized to zero"

def test_zero_delay_check():
    """Verify that step_jr raises error for zero delays."""
    n_node = 4
    n_time = 64
    dt = 0.1
    nt = 10
    n_batch = 1
    width = 8
    
    # Setup Conn with a zero delay
    from scipy import sparse
    W = sparse.eye(n_node, format='csr', dtype=np.float32)
    conn = tvbk.Conn(n_node, W.nnz)
    conn.weights[:] = W.data
    conn.indices[:] = W.indices
    conn.indptr[:] = W.indptr
    
    # Set one delay to 0
    D = np.ones(W.data.shape, dtype=np.uint32)
    D[0] = 0
    conn.idelays[:] = D
    
    # Setup other args
    cxs = tvbk.Cx8s(n_node, n_time, n_batch)
    x = np.zeros((n_batch, 6, n_node, width), dtype=np.float32)
    y = np.zeros_like(x)
    z = np.zeros((n_batch, 6, width), dtype=np.float32)
    p = np.zeros((n_batch, 14, width), dtype=np.float32) # Param shape check might need fix?
    # In previous fix I changed p_mono to (batch, 1, parm, width).
    # step_jr requires p.shape[1] == num_node OR 1.
    # If I pass (batch, parm, width), it has 3 dims.
    # Let's use correct 4 dims: (batch, 1, 14, width)
    p = np.zeros((n_batch, 1, 14, width), dtype=np.float32)
    
    seed = np.zeros((n_batch, width, 4), dtype=np.uint64)
    
    # Expect error
    with pytest.raises(RuntimeError, match="Conn contains zero delays"):
        tvbk.step_jr(cxs, conn, x, y, z, p, 0, nt, dt, seed)

if __name__ == "__main__":
    test_buffer_initialization()
    test_zero_delay_check()
    print("All safety tests passed!")
