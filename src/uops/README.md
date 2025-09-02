# uops

**uops** is a C library and Python interface for microbenchmarking, parallel execution, and compositional interpretation of simple operations ("micro-ops") on arrays and scalars. It provides primitives for scan, loop, sequence, parallel map, and a flexible interpreter for custom instruction sets, with a focus on benchmarking, threading, and extensibility.

## Features

- **C Library**: Implements scan, loop, sequence, parallel map (pmap), and a generic interpreter for custom instruction sets.
- **Interpreter API**: Compose and execute tagged instruction sequences (including loops, blocks, and math ops) using `run_interpreter()`.
- **Threading**: Uses a lightweight threading library (`tinycthread`) for parallel execution.
- **Microbenchmarking**: Includes scalar and vector operations for benchmarking math functions (e.g., `sinf`).
- **Python Interface**: The `uops_py` package exposes the C library to Python, supporting NumPy arrays and benchmarking via `pytest-benchmark`.
- **Extensible API**: Easily add new operations and compose them for benchmarking, parallel execution, or interpretation.

## Installation

### C Library

```sh
git clone https://github.com/maedoc/uops.git
cd uops
mkdir build && cd build
cmake ..
make
```

### C API

```c
#include "uops.h"

scan_t s = { .f = my_scan_fn, .x = input, .y = output, .n = N, .sz = sizeof(float) };
scan(s);

loop_t l = { .op = my_op, .arg = &data, .n = 1000 };
loop(l);

seq_t seq = { .ops = ops_array, .args = args_array, .n = num_ops };
seq(seq);

pmap_t p = { .ops = ops_array, .args = args_array, .n = num_ops, .num_threads = 4 };
scan(s);

// Interpreter API (new in this branch)
#include "interpreter.h"
instruction_t program[] = {
  { OP_SINF_SCALAR, .args.sinf_scalar = { .val_ptr = &x } },
  { OP_LOOP_START,  .args.loop_start = { .count = 10 } },
  // ... more instructions ...
  { OP_HALT }
};
run_interpreter(program, sizeof(program)/sizeof(program[0]));

loop_t l = { .op = my_op, .arg = &data, .n = 1000 };
loop(l);

seq_t seq = { .ops = ops_array, .args = args_array, .n = num_ops };
seq(seq);

pmap_t p = { .ops = ops_array, .args = args_array, .n = num_ops, .num_threads = 4 };
pmap(p);
```


### Python API

```python
import numpy as np
import ctypes
from uops_py import scan, scan_t, scan_fn, loop, loop_t, op_t, seq, seq_t, pmap, pmap_t, OK, create_float_array, get_pointer, get_int32_pointer

# --- Scan Example ---
def python_scan_op(scan_f):
  x_ptr = ctypes.cast(scan_f.x, ctypes.POINTER(ctypes.c_float))
  y_ptr = ctypes.cast(scan_f.y, ctypes.POINTER(ctypes.c_float))
  y_ptr[0] = np.sin(x_ptr[0])
  return OK

scan_op = scan_fn(python_scan_op)
x = create_float_array([1.0, 2.0, 3.0])
y = create_float_array([0.0, 0.0, 0.0])
args = scan_t(f=scan_op, c=None, x=get_pointer(x), y=get_pointer(y), sz=ctypes.sizeof(ctypes.c_float), n=3)
scan(args)
print(y[:])

# --- Loop Example ---
def python_loop_op(arg):
  value_ptr = ctypes.cast(arg, ctypes.POINTER(ctypes.c_float))
  value_ptr[0] *= 2.0
  return OK

loop_op = op_t(python_loop_op)
val = ctypes.c_float(1.0)
counter = ctypes.c_int32(0)
args = loop_t(op=loop_op, arg=ctypes.cast(ctypes.pointer(val), ctypes.c_void_p), inc=get_int32_pointer(counter), n=5)
loop(args)
print(val.value)

# --- Seq Example ---
def add_int_op(arg):
  ptr = ctypes.cast(arg, ctypes.POINTER(ctypes.c_int))
  ptr[0] += 5
  return OK
def multiply_float_op(arg):
  ptr = ctypes.cast(arg, ctypes.POINTER(ctypes.c_float))
  ptr[0] *= 2.0
  return OK
def print_string_op(arg):
  char_ptr = ctypes.cast(arg, ctypes.c_char_p)
  print(char_ptr.value.decode())
  return OK

ops_array = (op_t * 3)(op_t(add_int_op), op_t(multiply_float_op), op_t(print_string_op))
int_val = ctypes.c_int(10)
float_val = ctypes.c_float(3.14)
string_val = ctypes.create_string_buffer(b"Hello Sequence!")
args_array = (ctypes.c_void_p * 3)(ctypes.cast(ctypes.pointer(int_val), ctypes.c_void_p), ctypes.cast(ctypes.pointer(float_val), ctypes.c_void_p), ctypes.cast(string_val, ctypes.c_void_p))
seq_args = seq_t(ops=ops_array, args=args_array, n=3)
seq(seq_args)
print(int_val.value, float_val.value)

# --- Pmap Example ---
def work_op(arg):
  print("Work done!")
  return OK
ops_array = (op_t * 4)(*[op_t(work_op)] * 4)
args_array = (ctypes.c_void_p * 4)(*[None] * 4)
pmap_args = pmap_t(ops=ops_array, args=args_array, n=4, num_threads=2)
pmap(pmap_args)
```



## Code Style

- **Indentation**: 4 spaces per indentation level; no tabs.
- **Braces**: Opening braces on the same line for functions and control structures.
- **Function Definitions**: Use `INLINE static` for internal helper functions; public API functions use `UOPS_API`.
- **Validation**: All public and internal functions validate input arguments and return `ERR` on invalid input.
- **Early Exit**: Use early returns for error handling and invalid states.
- **Error Handling**: Return `ERR` for errors; use bitwise OR (`|=`) to accumulate error codes in loops.
- **Comments**: Use C-style `//` for inline comments and `/* ... */` for block comments. Brief doc comments are used for structs and functions.
- **Naming**: Lowercase with underscores for variables and functions (`scan_inline`, `loop_inline`, `work_f`). Structs and typedefs use `_t` suffix.
- **Memory Management**: Use `malloc`/`free` for dynamic allocations; always clean up resources.
- **Threading**: Use `tinycthread` for portable threading; thread entry functions return status codes.
- **Includes**: Standard headers first, then project headers.
- **Control Flow**: Prefer simple for/while loops; avoid complex nesting.
- **API Design**: All API functions take a single struct argument for extensibility.

## Authors

- Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>
