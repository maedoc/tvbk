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
    
    # Define heun_work_t structure
    class heun_work_t(ctypes.Structure):
        _fields_ = [
            ("n", ctypes.c_int),
            ("dt", ctypes.c_float),
            ("dx1", ctypes.POINTER(ctypes.c_float)),
            ("dx2", ctypes.POINTER(ctypes.c_float)),
            ("xi", ctypes.POINTER(ctypes.c_float)),
            ("x", ctypes.POINTER(ctypes.c_float))
        ]

    # Set up argument and return types
    lib.tvbk_heun_alloc.argtypes = [ctypes.c_int, ctypes.c_float]
    lib.tvbk_heun_alloc.restype = ctypes.POINTER(heun_work_t)
    
    lib.tvbk_heun_free.argtypes = [ctypes.POINTER(heun_work_t)]
    lib.tvbk_heun_free.restype = None
    
    lib.tvbk_model_stepn.argtypes = [ctypes.POINTER(heun_work_t)]
    lib.tvbk_model_stepn.restype = None
    
    # Create test data and work structs
    n = 32
    dt = 0.1
    x = np.random.rand(n).astype(np.float32)
    
    # Allocate work struct and set x pointer
    work = lib.tvbk_heun_alloc(n, dt)
    work.contents.x = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Benchmark
    ips = benchmark(lambda _: lib.tvbk_model_stepn(work), None)
    print(f"tvbk_model_stepn(n={n}): {ips:,.0f} iterations/second")
    
    # Clean up
    lib.tvbk_heun_free(work)

if __name__ == "__main__":
    main()
