import ctypes
import time
import numpy as np

def benchmark(func, x, iterations=100000, warmup=100):
    """Benchmark a function by running it multiple times."""
    # Warmup
    for _ in range(warmup):
        func(x)
    
    # Timed run
    start = time.perf_counter()
    for _ in range(iterations):
        func(x)
    duration = time.perf_counter() - start
    
    ips = iterations / duration
    return ips

def main():
    # Load the shared library
    lib = ctypes.CDLL('./foo.so')
    
    # Set up argument and return types
    lib.tvbk_model_step32.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')]
    lib.tvbk_model_step32.restype = None
    
    lib.tvbk_model_stepn.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
    ]
    lib.tvbk_model_stepn.restype = None
    
    # Create test data
    x32 = np.random.rand(32).astype(np.float32)
    xn = np.random.rand(128).astype(np.float32)
    
    # Benchmark step32
    ips32 = benchmark(lambda x: lib.tvbk_model_step32(x), x32)
    print(f"tvbk_model_step32: {ips32:,.0f} iterations/second")
    
    # Benchmark stepn with n=32
    ipsn = benchmark(lambda x: lib.tvbk_model_stepn(32, x), xn)
    print(f"tvbk_model_stepn(n=32): {ipsn:,.0f} iterations/second")

if __name__ == "__main__":
    main()
