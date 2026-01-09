
import jax
from jax import dtypes
from jax import core as jax_core
from jax.extend import core as extend_core
from jax.interpreters import xla, mlir 
import jax.numpy as jp
import numpy as np

from jax.experimental.x64_context import config
xla_client = config.xla_client

import tvbk.tvbk_ext as _ext

# Register the custom call target
_registrations = _ext.registrations()
for name, fn in _registrations.items():
    xla_client.register_custom_call_target(name, fn, platform="cpu")

coupling_p = extend_core.Primitive("coupling")

def coupling(buffer, weights, indices, indptr, idelays, t):
    """
    JAX-compatible coupling function using C++ kernel.
    buffer: (num_node, num_time, width=8) float32
    weights: (nnz) float32
    indices: (nnz) uint32/int32
    indptr: (num_node+1) uint32/int32
    idelays: (nnz) uint32/int32
    t: scalar integer
    
    Returns: cx (2, num_node, width=8) float32
    """
    return coupling_p.bind(buffer, weights, indices, indptr, idelays, t)

def coupling_abstract_eval(buffer, weights, indices, indptr, idelays, t):
    shape = buffer.shape
    if len(shape) == 3:
        num_node = shape[0]
        width = shape[2]
        return jax_core.ShapedArray((2, num_node, width), buffer.dtype)
    elif len(shape) == 4:
        # Batched input: (Batch, Node, Time, Width)
        batch_size = shape[0]
        num_node = shape[1]
        width = shape[3]
        return jax_core.ShapedArray((batch_size, 2, num_node, width), buffer.dtype)
    else:
        raise ValueError(f"Unsupported buffer rank: {len(shape)}")

def coupling_lowering_cpu(ctx, buffer, weights, indices, indptr, idelays, t):
    buffer_type = mlir.ir.RankedTensorType(buffer.type)
    buffer_shape = buffer_type.shape
    
    weights_type = mlir.ir.RankedTensorType(weights.type)
    num_nonzero = weights_type.shape[0]

    i32_type = mlir.ir.IntegerType.get_unsigned(32)
    def make_const(val):
        return mlir.ir.Operation.create(
            "stablehlo.constant",
            results=[mlir.ir.RankedTensorType.get([], i32_type)],
            attributes={"value": mlir.ir.DenseIntElementsAttr.get(
                np.array(val, dtype=np.uint32),
                type=mlir.ir.RankedTensorType.get([], i32_type)
            )}
        ).results[0]
    
    num_nonzero_op = make_const(num_nonzero)

    # Detect rank to choose kernel
    if len(buffer_shape) == 3:
        num_node = buffer_shape[0]
        num_time = buffer_shape[1]
        width = buffer_shape[2]
        num_node_op = make_const(num_node)
        num_time_op = make_const(num_time)
        
        target_name = "coupling_kernel_cpu"
        f32_type = mlir.ir.F32Type.get()
        out_type = mlir.ir.RankedTensorType.get([2, num_node, width], f32_type)
        operands = [buffer, weights, indices, indptr, idelays, t, num_node_op, num_time_op, num_nonzero_op]
        operand_layouts = [
            (2, 1, 0), 
            (0,), (0,), (0,), (0,), 
            (), (), (), () 
        ]
        result_layouts = [(2, 1, 0)]

    elif len(buffer_shape) == 4:
        batch_size = buffer_shape[0]
        num_node = buffer_shape[1]
        num_time = buffer_shape[2]
        width = buffer_shape[3]
        
        num_node_op = make_const(num_node)
        num_time_op = make_const(num_time)
        batch_size_op = make_const(batch_size)
        
        target_name = "coupling_batch_kernel_cpu"
        f32_type = mlir.ir.F32Type.get()
        # Output: (Batch, 2, Node, Width)
        out_type = mlir.ir.RankedTensorType.get([batch_size, 2, num_node, width], f32_type)
        operands = [buffer, weights, indices, indptr, idelays, t, num_node_op, num_time_op, num_nonzero_op, batch_size_op]
        # Layouts: (3, 2, 1, 0) for 4D
        operand_layouts = [
            (3, 2, 1, 0), # buffer
            (0,), (0,), (0,), (0,), # weights etc
            (), (), (), (), () # scalars
        ]
        result_layouts = [(3, 2, 1, 0)] # output
        
    else:
         raise ValueError(f"Unsupported buffer rank: {len(buffer_shape)}")

    # Helper for manual custom_call creation
    def manual_custom_call(target_name, result_types, operands, operand_layouts, result_layouts):
        element_type = mlir.ir.IndexType.get()
        
        def make_layout_attr(layout):
            if len(layout) == 0:
                # Rank 1 tensor of size 0 for scalar/empty layout
                return mlir.ir.DenseIntElementsAttr.get(
                    np.array([], dtype=np.int64),
                    type=mlir.ir.RankedTensorType.get([0], element_type)
                )
            return mlir.ir.DenseIntElementsAttr.get(
                np.array(layout, dtype=np.int64), 
                type=mlir.ir.RankedTensorType.get([len(layout)], element_type)
            )
        
        op_layouts_attr = mlir.ir.ArrayAttr.get([make_layout_attr(l) for l in operand_layouts])
        res_layouts_attr = mlir.ir.ArrayAttr.get([make_layout_attr(l) for l in result_layouts])
        
        attrs = {
            "call_target_name": mlir.ir.StringAttr.get(target_name),
            "operand_layouts": op_layouts_attr,
            "result_layouts": res_layouts_attr,
            "has_side_effect": mlir.ir.BoolAttr.get(False),
             "api_version": mlir.ir.IntegerAttr.get(
                mlir.ir.IntegerType.get_signless(32), 1
            )
        }
        
        return mlir.ir.Operation.create(
            "stablehlo.custom_call",
            results=result_types,
            operands=operands,
            attributes=attrs
        ).results

    # Call custom call
    return manual_custom_call(
        target_name,
        result_types=[out_type],
        operands=operands,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts
    )

mlir.register_lowering(coupling_p, coupling_lowering_cpu, platform="cpu")
coupling_p.def_abstract_eval(coupling_abstract_eval)

def coupling_batch_rule(args, axes):
    buffer, weights, indices, indptr, idelays, t = args
    bd_buffer, bd_weights, bd_indices, bd_indptr, bd_idelays, bd_t = axes
    
    # Check if only buffer is batched (Sim Batch case)
    if bd_buffer is not None and all(x is None for x in [bd_weights, bd_indices, bd_indptr, bd_idelays]):
        # t can be batched or not? Usually t is scalar shared.
        if bd_t is not None:
             # If t is batched, we technically could handle it if we passed t[] to kernel.
             # But our kernel takes scalar t.
             # Fallback to loop if t is batched.
             return jax.vmap(coupling_p.bind, axes)(*args), 0
        
        # Ensure buffer is (Batch, Node, Time, Width)
        # If axis is not 0, move it to 0.
        if bd_buffer != 0:
             buffer = jp.moveaxis(buffer, bd_buffer, 0)
        
        # Dispatch to coupling primitive with batched input
        # Note: We reuse the same primitive but handle standard 4D input in lowering.
        out = coupling_p.bind(buffer, weights, indices, indptr, idelays, t)
        return out, 0
        
    # Fallback for Mixed Batching or t batching
    return jax.vmap(coupling_p.bind, axes)(*args), 0

from jax.interpreters import batching
batching.primitive_batchers[coupling_p] = coupling_batch_rule

