import pytest
import ctypes
import math
import platform
import os
import numpy as np # Using numpy for easier array creation and comparison
from uops_py import (
    loop, loop_t, op_t, # Renamed loop_api to loop
    # Scalar ops (keep for reference or potential future tests)
    sinf_op_c_op_t, sinf_loop_args_t, sinf_loop_c,
    # Vector ops
    vector_op_args_t, sinf_vector_op_c_op_t, sinf_vector_op_c_ptr, # Added raw C func ptr
    sinf_vector_loop_args_t, sinf_vector_loop_c, sinf_vector_loop_composed_c, # Added composed version
    # Base types
    err_t, OK, ERR, size_t,
    # Interpreter types
    instruction_t, op_args_union_t, op_args_sinf_vector_t, op_args_loop_start_t, op_args_seq_t, op_args_block_t, # Added op_args_block_t
    OP_LOOP_START, OP_SINF_VECTOR, OP_LOOP_END, OP_HALT, OP_SEQ,
    OP_BLOCK, OP_RETURN, # Added OP_BLOCK, OP_RETURN
    run_interpreter,
    # Helper (though numpy handles array creation now)
    create_float_array, get_pointer
)

# Check if running in CI or explicitly disabled to skip potentially long benchmarks
# SKIP_BENCHMARKS = os.environ.get("CI", "false").lower() == "true" or \
#                   os.environ.get("SKIP_BENCHMARKS", "false").lower() == "true"
# benchmark_marker = pytest.mark.skipif(SKIP_BENCHMARKS, reason="Skipping benchmarks in CI or by request")
# For now, let's always run it unless explicitly skipped via pytest -m "not benchmark"
benchmark_marker = pytest.mark.benchmark

# --- Vector Numerical Equivalence Test ---

def test_sinf_vector_loop_numerical_equivalence():
    """
    Verifies that both vector looping methods produce the same numerical result.
    """
    vector_len = 32 # Intermediate vector size
    n_iterations = 100 # Apply sinf to the whole vector 100 times
    initial_val = 1.57 # Approx pi/2

    # Create initial data using numpy for convenience
    data1_np = np.full(vector_len, initial_val, dtype=np.float32) # For loop_api
    data2_np = np.full(vector_len, initial_val, dtype=np.float32) # For handwritten C loop
    data3_np = np.full(vector_len, initial_val, dtype=np.float32) # For composed C loop
    data4_np = np.full(vector_len, initial_val, dtype=np.float32) # For Python loop
    data5_np = np.full(vector_len, initial_val, dtype=np.float32) # For Interpreter loop (direct op)
    data6_np = np.full(vector_len, initial_val, dtype=np.float32) # For Interpreter block loop

    # Get ctypes pointers
    data1_ptr = data1_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    data2_ptr = data2_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    data3_ptr = data3_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    data4_ptr = data4_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    data5_ptr = data5_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    data6_ptr = data6_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # --- Method 1: Python-controlled loop via loop ---
    vec_args1 = vector_op_args_t(data=data1_ptr, len=vector_len)
    vec_args1_ptr = ctypes.pointer(vec_args1) # Need pointer for op_t arg
    counter1 = ctypes.c_int32(0)
    loop_args = loop_t(
        op=sinf_vector_op_c_op_t, # The vector op
        arg=ctypes.cast(vec_args1_ptr, ctypes.c_void_p), # Pass pointer to vector args
        inc=ctypes.pointer(counter1),
        n=n_iterations # loop iterates n_iterations times
    )
    result1 = loop(loop_args) # Use renamed function

    # --- Method 2: C-controlled loop via sinf_vector_loop_c ---
    vec_loop_args = sinf_vector_loop_args_t(
        data=data2_ptr,
        len=vector_len,
        iterations=n_iterations # Dedicated loop takes iterations directly
    )
    vec_loop_args_ptr = ctypes.pointer(vec_loop_args)
    result2 = sinf_vector_loop_c(vec_loop_args_ptr)

    # --- Method 3: C-composed loop via sinf_vector_loop_composed_c ---
    vec_loop_args_composed = sinf_vector_loop_args_t(
        data=data3_ptr,
        len=vector_len,
        iterations=n_iterations
    )
    vec_loop_args_composed_ptr = ctypes.pointer(vec_loop_args_composed)
    result3 = sinf_vector_loop_composed_c(vec_loop_args_composed_ptr)

    # --- Method 4: Python loop calling C vector op directly ---
    vec_args4 = vector_op_args_t(data=data4_ptr, len=vector_len)
    vec_args4_ptr = ctypes.pointer(vec_args4)
    vec_args4_void_p = ctypes.cast(vec_args4_ptr, ctypes.c_void_p)
    result4 = OK # Initialize result for the loop
    for _ in range(n_iterations):
        res_inner = sinf_vector_op_c_ptr(vec_args4_void_p)
        if res_inner != OK:
            result4 = ERR
            break # Exit loop on first error

    # --- Method 5: Interpreter loop ---
    # Define instructions
    instructions_list = [
        instruction_t(tag=OP_LOOP_START, args=op_args_union_t(loop_start=op_args_loop_start_t(count=n_iterations))),
        instruction_t(tag=OP_SINF_VECTOR, args=op_args_union_t(sinf_vector=op_args_sinf_vector_t(data=data5_ptr, len=vector_len))),
        instruction_t(tag=OP_LOOP_END, args=op_args_union_t()), # Explicitly init args
        instruction_t(tag=OP_HALT, args=op_args_union_t())      # Explicitly init args
    ]
    num_instructions = len(instructions_list)
    InstructionsArray5 = instruction_t * num_instructions
    instructions_arr5 = InstructionsArray5(*instructions_list)
    result5 = run_interpreter(instructions_arr5, num_instructions)

    # --- Method 6: Interpreter block containing the loop ---
    # Block instructions
    block_instructions_list = [
        instruction_t(tag=OP_LOOP_START, args=op_args_union_t(loop_start=op_args_loop_start_t(count=n_iterations))),
        instruction_t(tag=OP_SINF_VECTOR, args=op_args_union_t(sinf_vector=op_args_sinf_vector_t(data=data6_ptr, len=vector_len))),
        instruction_t(tag=OP_LOOP_END, args=op_args_union_t()),
        instruction_t(tag=OP_RETURN, args=op_args_union_t())
    ]
    block_num_instructions = len(block_instructions_list)
    BlockInstructionsArray = instruction_t * block_num_instructions
    block_instructions_arr = BlockInstructionsArray(*block_instructions_list)

    # Main instructions
    main_instructions_list = [
        instruction_t(tag=OP_BLOCK, args=op_args_union_t(block=op_args_block_t(instructions=block_instructions_arr, num_instructions=block_num_instructions))),
        instruction_t(tag=OP_HALT, args=op_args_union_t())
    ]
    main_num_instructions = len(main_instructions_list)
    MainInstructionsArray = instruction_t * main_num_instructions
    main_instructions_arr = MainInstructionsArray(*main_instructions_list)
    result6 = run_interpreter(main_instructions_arr, main_num_instructions)


    # --- Verification ---
    print(f"\nVerifying {n_iterations} vector sinf iterations (len={vector_len}):")
    assert result1 == OK, "Method 1 (loop) failed"
    assert result2 == OK, "Method 2 (sinf_vector_loop_c - handwritten) failed"
    assert result3 == OK, "Method 3 (sinf_vector_loop_composed_c) failed"
    assert result4 == OK, "Method 4 (Python loop calling C op) failed"
    assert result5 == OK, "Method 5 (Interpreter loop - direct op) failed"
    assert result6 == OK, "Method 6 (Interpreter block loop) failed"

    # Compare numpy arrays for numerical closeness
    assert np.allclose(data1_np, data2_np, rtol=1e-6), \
        f"Results differ between loop and handwritten C."
    assert np.allclose(data2_np, data3_np, rtol=1e-6), \
        f"Results differ between handwritten C and composed C."
    assert np.allclose(data3_np, data4_np, rtol=1e-6), \
        f"Results differ between composed C and Python loop."
    assert np.allclose(data4_np, data5_np, rtol=1e-6), \
        f"Results differ between Python loop and Interpreter loop (direct op)."
    assert np.allclose(data5_np, data6_np, rtol=1e-6), \
        f"Results differ between Interpreter loop (direct op) and Interpreter block loop."
    print("Vector numerical equivalence test passed for all methods.")


# --- Vector Benchmark Tests ---

@benchmark_marker
def test_sinf_vector_loop_api_benchmark(benchmark):
    """Benchmarks the loop_api method with a vector operation."""
    vector_len = 32 # Intermediate vector size
    n_iterations = 100 # Apply sinf to the whole vector 100 times
    initial_val = 1.57

    data_np = np.full(vector_len, initial_val, dtype=np.float32)
    data_ptr = data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    initial_data_copy = data_np.copy() # For resetting

    vec_args = vector_op_args_t(data=data_ptr, len=vector_len)
    vec_args_ptr = ctypes.pointer(vec_args)
    counter = ctypes.c_int32(0)
    loop_args = loop_t(
        op=sinf_vector_op_c_op_t,
        arg=ctypes.cast(vec_args_ptr, ctypes.c_void_p),
        inc=ctypes.pointer(counter),
        n=n_iterations
    )

    def setup():
        # Reset data array and counter before each benchmark round
        np.copyto(data_np, initial_data_copy)
        counter.value = 0

    # Benchmark the loop call
    result = benchmark(loop, loop_args, setup=setup) # Use renamed function
    assert result == OK


@benchmark_marker
def test_sinf_vector_loop_c_benchmark(benchmark):
    """Benchmarks the internal C vector loop method."""
    vector_len = 32 # Intermediate vector size
    n_iterations = 100 # Apply sinf to the whole vector 100 times
    initial_val = 1.57

    data_np = np.full(vector_len, initial_val, dtype=np.float32)
    data_ptr = data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    initial_data_copy = data_np.copy() # For resetting

    vec_loop_args = sinf_vector_loop_args_t(
        data=data_ptr,
        len=vector_len,
        iterations=n_iterations
    )
    vec_loop_args_ptr = ctypes.pointer(vec_loop_args)

    def setup():
        # Reset data array before each benchmark round
        np.copyto(data_np, initial_data_copy)

    # Benchmark the dedicated C vector loop call
    result = benchmark(sinf_vector_loop_c, vec_loop_args_ptr, setup=setup)
    assert result == OK


@benchmark_marker
def test_sinf_vector_loop_composed_c_benchmark(benchmark):
    """Benchmarks the composed C vector loop method."""
    vector_len = 32 # Intermediate vector size (Keep consistent)
    n_iterations = 100 # Apply sinf to the whole vector 100 times (Keep consistent)
    initial_val = 1.57

    data_np = np.full(vector_len, initial_val, dtype=np.float32)
    data_ptr = data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    initial_data_copy = data_np.copy() # For resetting

    vec_loop_args = sinf_vector_loop_args_t(
        data=data_ptr,
        len=vector_len,
        iterations=n_iterations
    )
    vec_loop_args_ptr = ctypes.pointer(vec_loop_args)

    def setup():
        # Reset data array before each benchmark round
        np.copyto(data_np, initial_data_copy)

    # Benchmark the composed C vector loop call
    result = benchmark(sinf_vector_loop_composed_c, vec_loop_args_ptr, setup=setup)
    assert result == OK


@benchmark_marker
def test_sinf_vector_python_loop_benchmark(benchmark):
    """Benchmarks a Python loop calling the C vector operation directly."""
    vector_len = 32 # Intermediate vector size
    n_iterations = 100 # Apply sinf to the whole vector 100 times
    initial_val = 1.57

    data_np = np.full(vector_len, initial_val, dtype=np.float32)
    data_ptr = data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    initial_data_copy = data_np.copy() # For resetting

    vec_args = vector_op_args_t(data=data_ptr, len=vector_len)
    vec_args_ptr = ctypes.pointer(vec_args)
    # Cast the specific pointer to void* for the C function call
    vec_args_void_p = ctypes.cast(vec_args_ptr, ctypes.c_void_p)

    def setup():
        # Reset data array before each benchmark round
        np.copyto(data_np, initial_data_copy)

    def run_python_loop(**kwargs): # Accept kwargs to work around pytest-benchmark issue
        for _ in range(n_iterations):
            result = sinf_vector_op_c_ptr(vec_args_void_p)
            if result != OK:
                 raise RuntimeError("C function sinf_vector_op_c_ptr failed")

    # Benchmark the function containing the Python loop
    benchmark(run_python_loop, setup=setup)
    # No explicit result check needed here as benchmark runs the function,
    # and run_python_loop raises RuntimeError on failure.


@benchmark_marker
def test_sinf_vector_interpreter_benchmark(benchmark):
    """Benchmarks the interpreter executing a loop of vector sinf operations."""
    vector_len = 32
    n_iterations = 100
    initial_val = 1.57

    data_np = np.full(vector_len, initial_val, dtype=np.float32)
    data_ptr = data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    initial_data_copy = data_np.copy()

    # Define instructions - create once
    instructions_list = [
        instruction_t(tag=OP_LOOP_START, args=op_args_union_t(loop_start=op_args_loop_start_t(count=n_iterations))),
        instruction_t(tag=OP_SINF_VECTOR, args=op_args_union_t(sinf_vector=op_args_sinf_vector_t(data=data_ptr, len=vector_len))),
        instruction_t(tag=OP_LOOP_END, args=op_args_union_t()), # Explicitly init args
        instruction_t(tag=OP_HALT, args=op_args_union_t())      # Explicitly init args
    ]
    num_instructions = len(instructions_list)
    InstructionsArray = instruction_t * num_instructions
    instructions_arr = InstructionsArray(*instructions_list)

    def setup():
        # Reset data array before each benchmark round
        np.copyto(data_np, initial_data_copy)

    # Benchmark the interpreter call
    result = benchmark(run_interpreter, instructions_arr, num_instructions, setup=setup)
    assert result == OK


@benchmark_marker
def test_sinf_vector_interpreter_block_loop_benchmark(benchmark):
    """Benchmarks the interpreter executing a block containing the loop."""
    vector_len = 32
    n_iterations = 100
    initial_val = 1.57

    data_np = np.full(vector_len, initial_val, dtype=np.float32)
    data_ptr = data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    initial_data_copy = data_np.copy()

    # --- Define Instructions (Create Once) ---
    # Block instructions (contains the loop)
    block_instructions_list = [
        instruction_t(tag=OP_LOOP_START, args=op_args_union_t(loop_start=op_args_loop_start_t(count=n_iterations))),
        instruction_t(tag=OP_SINF_VECTOR, args=op_args_union_t(sinf_vector=op_args_sinf_vector_t(data=data_ptr, len=vector_len))),
        instruction_t(tag=OP_LOOP_END, args=op_args_union_t()),
        instruction_t(tag=OP_RETURN, args=op_args_union_t())
    ]
    block_num_instructions = len(block_instructions_list)
    BlockInstructionsArray = instruction_t * block_num_instructions
    block_instructions_arr = BlockInstructionsArray(*block_instructions_list)

    # Main instructions (calls the block)
    main_instructions_list = [
        instruction_t(tag=OP_BLOCK, args=op_args_union_t(block=op_args_block_t(instructions=block_instructions_arr, num_instructions=block_num_instructions))),
        instruction_t(tag=OP_HALT, args=op_args_union_t())
    ]
    main_num_instructions = len(main_instructions_list)
    MainInstructionsArray = instruction_t * main_num_instructions
    main_instructions_arr = MainInstructionsArray(*main_instructions_list)
    # --- End Instruction Definition ---

    def setup():
        # Reset data array before each benchmark round
        np.copyto(data_np, initial_data_copy)

    # Benchmark the interpreter call executing the main instructions
    result = benchmark(run_interpreter, main_instructions_arr, main_num_instructions, setup=setup)
    assert result == OK


# Removing test_sinf_vector_interpreter_seq_benchmark as it becomes identical to
# test_sinf_vector_interpreter_benchmark when following the recommendation to
# use native opcodes directly inside loops instead of OP_SEQ for this purpose.
# Keeping OP_SEQ definition for potential use cases involving external C functions.
