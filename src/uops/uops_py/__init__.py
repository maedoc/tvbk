import ctypes
import sys
import os
import platform
import glob

for f in glob.glob('**/*uops*', recursive=True):
    print('***   ', f, file=sys.stderr)

# --- Library Loading ---

def find_library(lib_name_base):
    """Finds the shared library, handling platform differences."""
    lib_name = ""
    system = platform.system()
    if system == "Linux":
        lib_name = f"lib{lib_name_base}.so"
    elif system == "Darwin": # macOS
        lib_name = f"lib{lib_name_base}.dylib"
    elif system == "Windows":
        lib_name = f"{lib_name_base}.dll"
    else:
        raise OSError(f"Unsupported platform: {system}")

    # Look relative to this __init__.py file's location first
    package_dir = os.path.dirname(os.path.abspath(__file__))

    # Search paths relative to the installed package or development structure
    search_paths = [
        package_dir,
        os.path.join(package_dir, ".."),
        os.getcwd(),
        os.path.join(package_dir, "../buildlib"),        # Common build dir relative to package src
    ]

    print(f"Searching for library '{lib_name}' in paths:") # Debug print
    for path in search_paths:
        lib_path = os.path.join(path, lib_name)
        print(f"  Checking: {lib_path}") # Debug print
        if os.path.exists(lib_path):
            print(f"Found library at: {lib_path}")
            try:
                return ctypes.CDLL(lib_path)
            except OSError as e:
                print(f"Found library at {lib_path} but failed to load: {e}", file=sys.stderr)
                # Continue searching other paths

    # Fallback: try loading directly (might work if in LD_LIBRARY_PATH / PATH etc.)
    print(f"Library not found in specific paths. Trying direct load of '{lib_name}'...")
    try:
        # On Windows, CDLL might find DLLs in the PATH or application directory
        # On Linux/macOS, it might find libs in standard locations or LD_LIBRARY_PATH
        return ctypes.CDLL(lib_name)
    except OSError as e_fallback:
        print(f"Direct load failed: {e_fallback}", file=sys.stderr)
        raise OSError(
            f"Could not find or load library '{lib_name}'. "
            f"Checked paths: {search_paths}. "
            f"Ensure the compiled C library is accessible, either next to the installed "
            f"uops_py package, in a standard system library path, or by setting "
            f"LD_LIBRARY_PATH (Linux/macOS) or PATH (Windows)."
        ) from e_fallback

lib = find_library("uops")

# Basic types
err_t = ctypes.c_int # enum err { OK, ERR } -> typically int
OK = 0
ERR = 1
size_t = ctypes.c_size_t

# --- Scan API Types ---

# typedef struct scan_f { void*c,*x,*y; } scan_f_t;
class scan_f_t(ctypes.Structure):
    _fields_ = [
        ("c", ctypes.c_void_p),
        ("x", ctypes.c_void_p),
        ("y", ctypes.c_void_p),
    ]

# typedef err_t(*scan_fn)(scan_f_t);
scan_fn = ctypes.CFUNCTYPE(err_t, scan_f_t)

# typedef struct scan { ... } scan_t;
class scan_t(ctypes.Structure):
    _fields_ = [
        ("f", scan_fn),
        ("c", ctypes.c_void_p),
        ("x", ctypes.c_void_p),
        ("y", ctypes.c_void_p),
        ("sz", size_t),
        ("n", size_t),
    ]

# err_t scan(scan_t s);
scan = lib.scan
scan.argtypes = [scan_t]
scan.restype = err_t

# --- Loop API Types ---

# typedef err_t(*op_t)(void*);
op_t = ctypes.CFUNCTYPE(err_t, ctypes.c_void_p)

# typedef struct loop { ... } loop_t;
class loop_t(ctypes.Structure):
     _fields_ = [
        ("op", op_t),
        ("arg", ctypes.c_void_p),
        ("inc", ctypes.POINTER(ctypes.c_int32)),
        ("n", size_t),
    ]

# err_t loop(loop_t l);
loop = lib.loop
loop.argtypes = [loop_t]
loop.restype = err_t


# --- Seq API Types ---

# typedef struct seq { ... } seq_t;
class seq_t(ctypes.Structure):
    _fields_ = [
        ("ops", ctypes.POINTER(op_t)),
        ("args", ctypes.POINTER(ctypes.c_void_p)),
        ("n", size_t),
    ]

# err_t seq(seq_t s);
seq = lib.seq
seq.argtypes = [seq_t]
seq.restype = err_t


# --- Pmap API Types ---

# typedef struct pmap { ... } pmap_t;
class pmap_t(ctypes.Structure):
    _fields_ = [
        ("ops", ctypes.POINTER(op_t)),
        ("args", ctypes.POINTER(ctypes.c_void_p)),
        ("n", size_t),
        ("num_threads", size_t), # 0 means auto/default
    ]

# err_t pmap(pmap_t p);
pmap = lib.pmap
pmap.argtypes = [pmap_t]
pmap.restype = err_t


# --- Specific Ops Interface ---

# err_t sinf_op_c(void* arg);
sinf_op_c_ptr = lib.sinf_op_c
sinf_op_c_ptr.argtypes = [ctypes.c_void_p]
sinf_op_c_ptr.restype = err_t
sinf_op_c_op_t = op_t(sinf_op_c_ptr)


# typedef struct sinf_loop_args { ... } sinf_loop_args_t;
class sinf_loop_args_t(ctypes.Structure):
    _fields_ = [
        ("val_ptr", ctypes.POINTER(ctypes.c_float)),
        ("n", size_t),
    ]

# err_t sinf_loop_c(sinf_loop_args_t* args);
sinf_loop_c = lib.sinf_loop_c
sinf_loop_c.argtypes = [ctypes.POINTER(sinf_loop_args_t)]
sinf_loop_c.restype = err_t


# --- Vector Ops Interface ---

# typedef struct vector_op_args { ... } vector_op_args_t;
class vector_op_args_t(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("len", size_t),
    ]

# err_t sinf_vector_op_c(void* arg);
sinf_vector_op_c_ptr = lib.sinf_vector_op_c
sinf_vector_op_c_ptr.argtypes = [ctypes.c_void_p] # op_t takes void*
sinf_vector_op_c_ptr.restype = err_t
sinf_vector_op_c_op_t = op_t(sinf_vector_op_c_ptr)


# typedef struct sinf_vector_loop_args { ... } sinf_vector_loop_args_t;
class sinf_vector_loop_args_t(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("len", size_t),
        ("iterations", size_t),
    ]

# err_t sinf_vector_loop_c(sinf_vector_loop_args_t* args);
sinf_vector_loop_c = lib.sinf_vector_loop_c
sinf_vector_loop_c.argtypes = [ctypes.POINTER(sinf_vector_loop_args_t)]
sinf_vector_loop_c.restype = err_t

# err_t sinf_vector_loop_composed_c(sinf_vector_loop_args_t* args);
sinf_vector_loop_composed_c = lib.sinf_vector_loop_composed_c
sinf_vector_loop_composed_c.argtypes = [ctypes.POINTER(sinf_vector_loop_args_t)]
sinf_vector_loop_composed_c.restype = err_t


# --- Interpreter API Types ---

# Constants for op_type_t enum
OP_NOOP        = 0
OP_HALT        = 1
OP_SINF_SCALAR = 2
OP_SINF_VECTOR = 3
OP_LOOP_START  = 4
OP_LOOP_END    = 5
OP_SEQ         = 6
OP_BLOCK       = 7 # Added OP_BLOCK constant
OP_RETURN      = 8 # Added OP_RETURN constant

# typedef struct op_args_sinf_scalar { ... } op_args_sinf_scalar_t;
class op_args_sinf_scalar_t(ctypes.Structure):
    _fields_ = [
        ("val_ptr", ctypes.POINTER(ctypes.c_float)),
    ]

# typedef struct op_args_sinf_vector { ... } op_args_sinf_vector_t;
class op_args_sinf_vector_t(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("len", size_t),
    ]

# typedef struct op_args_loop_start { ... } op_args_loop_start_t;
class op_args_loop_start_t(ctypes.Structure):
    _fields_ = [
        ("count", size_t),
    ]

# typedef struct op_args_seq { ... } op_args_seq_t;
class op_args_seq_t(ctypes.Structure):
     _fields_ = [
        ("ops", ctypes.POINTER(op_t)),
        ("args", ctypes.POINTER(ctypes.c_void_p)),
        ("n", size_t),
    ]

# Forward declaration for op_args_block_t
class instruction_t(ctypes.Structure): pass

# typedef struct op_args_block { ... } op_args_block_t;
class op_args_block_t(ctypes.Structure):
     _fields_ = [
        ("instructions", ctypes.POINTER(instruction_t)),
        ("num_instructions", size_t),
    ]

# typedef union op_args_union { ... } op_args_union_t;
class op_args_union_t(ctypes.Union):
    _fields_ = [
        ("sinf_scalar", op_args_sinf_scalar_t),
        ("sinf_vector", op_args_sinf_vector_t),
        ("loop_start", op_args_loop_start_t),
        ("seq", op_args_seq_t),
        ("block", op_args_block_t), # Added block field
        # Add other arg structs here if needed
    ]

# Actual definition of instruction_t (after op_args_block_t is defined)
instruction_t._fields_ = [
    ("tag", ctypes.c_int), # op_type_t enum maps to int
    ("args", op_args_union_t),
]

# err_t run_interpreter(instruction_t* instructions, size_t num_instructions);
run_interpreter = lib.run_interpreter
run_interpreter.argtypes = [ctypes.POINTER(instruction_t), size_t]
run_interpreter.restype = err_t


# --- Helper Functions ---

def create_float_array(data):
    """Creates a ctypes array of floats from a Python list."""
    return (ctypes.c_float * len(data))(*data)

def get_pointer(ctypes_array):
    """Gets a void pointer to the start of a ctypes array."""
    return ctypes.cast(ctypes_array, ctypes.c_void_p)

def get_int32_pointer(ctypes_int):
    """Gets a pointer to a ctypes int32."""
    # Ensure it's a ctypes type before taking pointer
    if not isinstance(ctypes_int, ctypes._SimpleCData):
         raise TypeError("Input must be a ctypes integer type (e.g., c_int32)")
    return ctypes.pointer(ctypes_int)
