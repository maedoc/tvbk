import unittest
import ctypes
import math
import platform
import threading # For pmap test shared data (lock needed)
import os # For skipping tests if lib not found

# Import from the package 'uops_py' now
from uops_py import (
    scan, scan_t, scan_f_t, scan_fn,
    loop, loop_t, op_t,
    seq, seq_t,
    pmap, pmap_t,
    err_t, OK, ERR, size_t,
    create_float_array, get_pointer, get_int32_pointer,
    lib # Import lib to check if it loaded
)

# --- Skip tests if C library wasn't loaded ---
# unittest.skipIf decorator needs a boolean condition and a reason
skip_reason = "uops C library not loaded or found"
skip_if_lib_missing = unittest.skipIf(lib is None, skip_reason)

# --- Scan Test ---

# Check if scan_fn is defined (it won't be if lib is None)
if scan_fn:
    @scan_fn
    def python_scan_op(scan_f):
        try:
            x_ptr = ctypes.cast(scan_f.x, ctypes.POINTER(ctypes.c_float))
            y_ptr = ctypes.cast(scan_f.y, ctypes.POINTER(ctypes.c_float))
            y_ptr[0] = math.sin(x_ptr[0])
            return OK
        except Exception as e:
            print(f"Error in python_scan_op: {e}")
            return ERR
else:
    # Define a dummy function if scan_fn is None, tests using it will be skipped
    def python_scan_op(*args, **kwargs):
        raise RuntimeError(skip_reason)


# --- Loop/Seq/Pmap Test Ops ---

# Check if op_t is defined
if op_t:
    @op_t
    def python_loop_op(arg_ptr):
        try:
            value_ptr = ctypes.cast(arg_ptr, ctypes.POINTER(ctypes.c_float))
            value_ptr[0] *= 1.1
            return OK
        except Exception as e:
            print(f"Error in python_loop_op: {e}")
            return ERR

    @op_t
    def python_add_int_op(arg):
        try:
            ptr = ctypes.cast(arg, ctypes.POINTER(ctypes.c_int))
            ptr[0] += 5
            return OK
        except Exception as e:
            print(f"Error in python_add_int_op: {e}")
            return ERR

    @op_t
    def python_multiply_float_op(arg):
        try:
            ptr = ctypes.cast(arg, ctypes.POINTER(ctypes.c_float))
            ptr[0] *= 2.0
            return OK
        except Exception as e:
            print(f"Error in python_multiply_float_op: {e}")
            return ERR

    @op_t
    def python_print_string_op(arg):
        try:
            char_ptr = ctypes.cast(arg, ctypes.c_char_p)
            print(f"python_print_string_op: {char_ptr.value.decode()}")
            return OK
        except Exception as e:
            print(f"Error in python_print_string_op: {e}")
            return ERR

    # Shared data structure and lock for pmap test
    pmap_shared_data = {
        "counter": 0,
        "lock": threading.Lock()
    }

    @op_t
    def python_work_op(arg):
        try:
            with pmap_shared_data["lock"]:
                pmap_shared_data["counter"] += 1
            return OK
        except Exception as e:
            print(f"Error in python_work_op: {e}")
            return ERR

    @op_t
    def python_fail_op(arg):
        try:
            return ERR # Signal failure
        except Exception as e:
            print(f"Error in python_fail_op: {e}")
            return ERR
else:
    # Define dummy functions if op_t is None, tests using them will be skipped
    def python_loop_op(*args, **kwargs): raise RuntimeError(skip_reason)
    def python_add_int_op(*args, **kwargs): raise RuntimeError(skip_reason)
    def python_multiply_float_op(*args, **kwargs): raise RuntimeError(skip_reason)
    def python_print_string_op(*args, **kwargs): raise RuntimeError(skip_reason)
    def python_work_op(*args, **kwargs): raise RuntimeError(skip_reason)
    def python_fail_op(*args, **kwargs): raise RuntimeError(skip_reason)
    pmap_shared_data = None # No shared data needed if ops are dummies


# --- Test Classes ---

@skip_if_lib_missing
class TestUopsScanLoop(unittest.TestCase):

    def test_scan_sin(self):
        print("\nRunning Python Scan Test...")
        n = 8
        x_data = [float(i + 1) for i in range(n)]
        y_data = [0.0] * n
        x_arr = create_float_array(x_data)
        y_arr = create_float_array(y_data)
        x_ptr = get_pointer(x_arr)
        y_ptr = get_pointer(y_arr)
        init_val = ctypes.c_float(0.0)
        init_ptr = ctypes.cast(ctypes.pointer(init_val), ctypes.c_void_p)

        args = scan_t(
            f=python_scan_op,
            c=init_ptr, x=x_ptr, y=y_ptr,
            sz=ctypes.sizeof(ctypes.c_float), n=n
        )
        result = scan(args)
        self.assertEqual(result, OK, "scan returned ERR")

        expected_y = [math.sin(x) for x in x_data]
        for i in range(n):
            self.assertAlmostEqual(y_arr[i], expected_y[i], places=6)
        print("Scan test passed!")

        # Test chaining
        print("\nRunning Python Scan Test (Chained)...")
        x_arr_2 = create_float_array([0.0] * n)
        x_ptr_2 = get_pointer(x_arr_2)
        args_2 = scan_t(
            f=python_scan_op, c=init_ptr, x=y_ptr, y=x_ptr_2,
            sz=ctypes.sizeof(ctypes.c_float), n=n
        )
        result_2 = scan(args_2)
        self.assertEqual(result_2, OK, "scan (chained) returned ERR")

        expected_x2 = [math.sin(y) for y in y_arr]
        for i in range(n):
             self.assertAlmostEqual(x_arr_2[i], expected_x2[i], places=6)
        print("Scan test (chained) passed!")


    def test_loop_multiply(self):
        print("\nRunning Python Loop Test...")
        loop_counter = ctypes.c_int32(0)
        op_value = ctypes.c_float(1.0)
        num_loops = 10

        print(f"Initial loop_counter: {loop_counter.value}")
        print(f"Initial op_value: {op_value.value}")

        # Use the helper for pointer creation
        loop_counter_ptr = get_int32_pointer(loop_counter)
        op_value_ptr = ctypes.cast(ctypes.pointer(op_value), ctypes.c_void_p)

        args = loop_t(
            op=python_loop_op,
            arg=op_value_ptr,
            inc=loop_counter_ptr,
            n=num_loops
        )
        result = loop(args)

        print(f"Final loop_counter: {loop_counter.value}")
        print(f"Final op_value: {op_value.value}")
        print(f"loop result: {'OK' if result == OK else 'ERR'}")

        expected_op_value = 1.0 * math.pow(1.1, float(num_loops))
        self.assertEqual(result, OK, "loop returned ERR")
        self.assertEqual(loop_counter.value, num_loops)
        self.assertAlmostEqual(op_value.value, expected_op_value, places=6)
        print("Loop test passed!")


@skip_if_lib_missing
class TestUopsSeqPmap(unittest.TestCase):

     def test_seq_basic(self):
        print("\nRunning Python Seq Test...")
        int_val = ctypes.c_int(10)
        float_val = ctypes.c_float(3.14)
        string_val_py = "Hello Sequence!"
        string_val = ctypes.create_string_buffer(string_val_py.encode('utf-8'))

        print(f"Initial int_val: {int_val.value}")
        print(f"Initial float_val: {float_val.value}")
        print(f"Initial string_val: {string_val.value.decode()}")

        ops_array = (op_t * 3)(
            python_add_int_op,
            python_multiply_float_op,
            python_print_string_op
        )
        args_array = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.pointer(int_val), ctypes.c_void_p),
            ctypes.cast(ctypes.pointer(float_val), ctypes.c_void_p),
            ctypes.cast(string_val, ctypes.c_void_p)
        )
        seq_args = seq_t(ops=ops_array, args=args_array, n=3)
        result = seq(seq_args)

        print(f"Final int_val: {int_val.value}")
        print(f"Final float_val: {float_val.value}")
        print(f"Final string_val: {string_val.value.decode()}")
        print(f"seq result: {'OK' if result == OK else 'ERR'}")

        self.assertEqual(result, OK, "seq returned ERR")
        self.assertEqual(int_val.value, 15)
        self.assertAlmostEqual(float_val.value, 6.28, places=6)
        self.assertEqual(string_val.value.decode(), string_val_py)
        print("Seq test passed!")


     def test_pmap_basic_success(self):
        print("\nRunning Python Pmap Success Test...")
        num_tasks = 8
        # Reset shared counter safely
        if pmap_shared_data:
            with pmap_shared_data["lock"]:
                 pmap_shared_data["counter"] = 0
            initial_counter = pmap_shared_data['counter']
        else:
            initial_counter = -1 # Should not happen if test runs
        print(f"Initial counter: {initial_counter}")

        ops_array = (op_t * num_tasks)(*[python_work_op] * num_tasks)
        args_array = (ctypes.c_void_p * num_tasks)(*[None] * num_tasks)
        pmap_args = pmap_t(ops=ops_array, args=args_array, n=num_tasks, num_threads=0)
        result = pmap(pmap_args)

        if pmap_shared_data:
            final_counter = pmap_shared_data['counter']
        else:
             final_counter = -1 # Should not happen
        print(f"Final counter: {final_counter}")
        print(f"pmap result: {'OK' if result == OK else 'ERR'}")

        self.assertEqual(result, OK, "pmap returned ERR")
        self.assertEqual(final_counter, num_tasks)
        print("Pmap success test passed!")


     def test_pmap_failure(self):
        print("\nRunning Python Pmap Failure Test...")
        num_tasks = 8
        fail_task_index = 3

        if pmap_shared_data:
            with pmap_shared_data["lock"]:
                 pmap_shared_data["counter"] = 0
            initial_counter = pmap_shared_data['counter']
        else:
            initial_counter = -1
        print(f"Initial counter: {initial_counter}")

        ops_list = [python_work_op] * num_tasks
        ops_list[fail_task_index] = python_fail_op
        ops_array = (op_t * num_tasks)(*ops_list)
        args_array = (ctypes.c_void_p * num_tasks)(*[None] * num_tasks)
        pmap_args = pmap_t(ops=ops_array, args=args_array, n=num_tasks, num_threads=0)
        result = pmap(pmap_args)

        if pmap_shared_data:
            final_counter = pmap_shared_data['counter']
        else:
            final_counter = -1
        print(f"Final counter: {final_counter}")
        print(f"pmap result: {'OK' if result == OK else 'ERR'}")

        self.assertEqual(result, ERR, "pmap did not return ERR on task failure")
        print(f"Pmap failure test passed (pmap returned ERR as expected). Final counter: {final_counter}")


if __name__ == '__main__':
    # This allows running the tests directly via 'python -m uops_py.tests.test_uops'
    # It will also be discovered by 'python -m unittest discover ...'
    unittest.main()
