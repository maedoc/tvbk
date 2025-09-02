#!/usr/bin/env python

import sys
# Assume foo_pb2.py is generated and in the python path
try:
    import foo_pb2
except ImportError:
    print("Error: foo_pb2.py not found. Please compile foo.proto first.", file=sys.stderr)
    print("Command: protoc --python_out=. foo.proto", file=sys.stderr)
    sys.exit(1)

from google.protobuf import text_format

# --- Helper function to create DataRef messages ---
def ref(name):
    return foo_pb2.DataRef(name=name)

# --- Define names for data buffers and scalars ---
# These names correspond to variables/buffers in the execution environment
X_IN = "x_in"
Z_OUT = "z_out"
TEMP1 = "temp1_vec" # Vector accumulator for sum
TEMP2 = "temp2_vec" # Vector accumulator for sum_sq / stddev fill
TEMP3 = "temp3_vec" # Vector intermediate for x^2 / (x-mean)

N_FLOAT = "N_float_scalar"
PARTIAL_SUM = "partial_sum_scalar"
PARTIAL_SUM_SQ = "partial_sum_sq_scalar"
MEAN = "mean_scalar"
MEAN_X_SQ = "mean_x_sq_scalar"
VARIANCE = "variance_scalar"
STDDEV = "stddev_scalar"
MEAN_SQUARED = "mean_squared_scalar" # Temp for variance calc

# --- Constants (assuming these are known when generating the sequence) ---
VECTOR_SIZE = 8 # Must match C code
TEST_N = VECTOR_SIZE * 10 # Example N, matching C code for demonstration
NUM_CHUNKS = TEST_N // VECTOR_SIZE # Calculate the actual number of chunks
# NUM_CHUNKS_VAR_NAME = "num_chunks_var" # Removed - Protobuf needs an integer count
CHUNK_BYTE_INC = VECTOR_SIZE * 4 # sizeof(float) = 4

# ========================================================================
# Phase 1: Calculate Mean and Mean of Squares
# ========================================================================
phase1_loop_body_seq = foo_pb2.InstructionSequence(instructions=[
    # Instruction 0: temp3 = x_chunk * x_chunk
    foo_pb2.Instruction(mul_vec8_args=foo_pb2.OpVec8Args(
        dst_data=ref(TEMP3), src1_data=ref(X_IN), src2_data=ref(X_IN)
    )),
    # Instruction 1: temp1 = temp1 + x_chunk (accumulate sum)
    foo_pb2.Instruction(add_vec8_args=foo_pb2.OpVec8Args(
        dst_data=ref(TEMP1), src1_data=ref(TEMP1), src2_data=ref(X_IN)
    )),
    # Instruction 2: temp2 = temp2 + temp3 (accumulate sum_sq)
    foo_pb2.Instruction(add_vec8_args=foo_pb2.OpVec8Args(
        dst_data=ref(TEMP2), src1_data=ref(TEMP2), src2_data=ref(TEMP3)
    )),
    # Instruction 3: Increment src1_data of the MUL instruction (instr 0)
    foo_pb2.Instruction(pinc_args=foo_pb2.OpPincArgs(
        target_instruction_index=0, target_argument_name="src1_data", increment_bytes=CHUNK_BYTE_INC
    )),
    # Instruction 4: Increment src2_data of the MUL instruction (instr 0)
    foo_pb2.Instruction(pinc_args=foo_pb2.OpPincArgs(
        target_instruction_index=0, target_argument_name="src2_data", increment_bytes=CHUNK_BYTE_INC
    )),
    # Instruction 5: Increment src2_data of the first ADD instruction (instr 1)
    foo_pb2.Instruction(pinc_args=foo_pb2.OpPincArgs(
        target_instruction_index=1, target_argument_name="src2_data", increment_bytes=CHUNK_BYTE_INC
    )),
    # Instruction 6: Return
    foo_pb2.Instruction(return_args=foo_pb2.OpReturnArgs())
])

phase1_seq = foo_pb2.InstructionSequence(instructions=[
    # Zero sum accumulator (temp1)
    foo_pb2.Instruction(zero_vec8_args=foo_pb2.OpVec8Args(dst_data=ref(TEMP1))),
    # Zero sum_sq accumulator (temp2)
    foo_pb2.Instruction(zero_vec8_args=foo_pb2.OpVec8Args(dst_data=ref(TEMP2))),
    # Loop
    # NOTE: Using the calculated NUM_CHUNKS integer value.
    #       The C interpreter will use this value directly.
    foo_pb2.Instruction(loop_start_args=foo_pb2.OpLoopStartArgs(count=NUM_CHUNKS)),
    foo_pb2.Instruction(block_args=foo_pb2.OpBlockArgs(sequence=phase1_loop_body_seq)),
    foo_pb2.Instruction(loop_end_args=foo_pb2.OpLoopEndArgs()),
    # Sum the final sum accumulator (temp1) -> partial_sum
    foo_pb2.Instruction(sum_vec8_args=foo_pb2.OpSumVec8Args(
        src1_data=ref(TEMP1), result_ref=ref(PARTIAL_SUM)
    )),
    # Sum the final sum_sq accumulator (temp2) -> partial_sum_sq
    foo_pb2.Instruction(sum_vec8_args=foo_pb2.OpSumVec8Args(
        src1_data=ref(TEMP2), result_ref=ref(PARTIAL_SUM_SQ)
    )),
    # Divide sum by N to get mean (store result back in partial_sum)
    foo_pb2.Instruction(div_scalar_args=foo_pb2.OpScalarBinaryArgs(
        dst_ref=ref(PARTIAL_SUM), src_ref=ref(N_FLOAT)
    )),
    # Divide sum_sq by N to get mean_x_sq (store result back in partial_sum_sq)
    foo_pb2.Instruction(div_scalar_args=foo_pb2.OpScalarBinaryArgs(
        dst_ref=ref(PARTIAL_SUM_SQ), src_ref=ref(N_FLOAT)
    )),
    # Store results into final mean/mean_x_sq variables
    foo_pb2.Instruction(copy_scalar_args=foo_pb2.OpScalarUnaryArgs(
        dst_ref=ref(MEAN), src_ref=ref(PARTIAL_SUM)
    )),
    foo_pb2.Instruction(copy_scalar_args=foo_pb2.OpScalarUnaryArgs(
        dst_ref=ref(MEAN_X_SQ), src_ref=ref(PARTIAL_SUM_SQ)
    )),
    # Return from Phase 1 block
    foo_pb2.Instruction(return_args=foo_pb2.OpReturnArgs())
])


# ========================================================================
# Phase 2: Calculate Variance = mean_x_sq - (mean * mean)
# ========================================================================
phase2_seq = foo_pb2.InstructionSequence(instructions=[
    # mean_squared = mean
    foo_pb2.Instruction(copy_scalar_args=foo_pb2.OpScalarUnaryArgs(
        dst_ref=ref(MEAN_SQUARED), src_ref=ref(MEAN)
    )),
    # mean_squared = mean_squared * mean (in-place)
    foo_pb2.Instruction(mul_scalar_args=foo_pb2.OpScalarBinaryArgs(
        dst_ref=ref(MEAN_SQUARED), src_ref=ref(MEAN)
    )),
    # variance = mean_x_sq
    foo_pb2.Instruction(copy_scalar_args=foo_pb2.OpScalarUnaryArgs(
        dst_ref=ref(VARIANCE), src_ref=ref(MEAN_X_SQ)
    )),
    # variance = variance - mean_squared (in-place)
    foo_pb2.Instruction(sub_scalar_args=foo_pb2.OpScalarBinaryArgs(
        dst_ref=ref(VARIANCE), src_ref=ref(MEAN_SQUARED)
    )),
    # Return from Phase 2 block
    foo_pb2.Instruction(return_args=foo_pb2.OpReturnArgs())
])

# ========================================================================
# Phase 3: Calculate Standard Deviation = sqrt(variance)
# ========================================================================
phase3_seq = foo_pb2.InstructionSequence(instructions=[
    # stddev = sqrt(variance)
    foo_pb2.Instruction(sqrt_scalar_args=foo_pb2.OpScalarUnaryArgs(
        dst_ref=ref(STDDEV), src_ref=ref(VARIANCE)
    )),
    # Return from Phase 3 block
    foo_pb2.Instruction(return_args=foo_pb2.OpReturnArgs())
])

# ========================================================================
# Phase 4: Calculate Z-Scores = (x - mean) / stddev
# ========================================================================
phase4_loop_body_seq = foo_pb2.InstructionSequence(instructions=[
    # Instruction 0: temp3 = x_chunk - temp1 (mean_vec)
    foo_pb2.Instruction(sub_vec8_args=foo_pb2.OpVec8Args(
        dst_data=ref(TEMP3), src1_data=ref(X_IN), src2_data=ref(TEMP1)
    )),
    # Instruction 1: z_chunk = temp3 / temp2 (stddev_vec)
    foo_pb2.Instruction(div_vec8_args=foo_pb2.OpVec8Args(
        dst_data=ref(Z_OUT), src1_data=ref(TEMP3), src2_data=ref(TEMP2)
    )),
    # Instruction 2: Increment src1_data of the SUB instruction (instr 0)
    foo_pb2.Instruction(pinc_args=foo_pb2.OpPincArgs(
        target_instruction_index=0, target_argument_name="src1_data", increment_bytes=CHUNK_BYTE_INC
    )),
    # Instruction 3: Increment dst_data of the DIV instruction (instr 1)
    foo_pb2.Instruction(pinc_args=foo_pb2.OpPincArgs(
        target_instruction_index=1, target_argument_name="dst_data", increment_bytes=CHUNK_BYTE_INC
    )),
    # Instruction 4: Return
    foo_pb2.Instruction(return_args=foo_pb2.OpReturnArgs())
])

phase4_seq = foo_pb2.InstructionSequence(instructions=[
    # temp1 = mean (scalar fill)
    foo_pb2.Instruction(fill_vec8_args=foo_pb2.OpFillVec8Args(
        dst_data=ref(TEMP1), scalar_ref=ref(MEAN)
    )),
    # temp2 = stddev (scalar fill)
    foo_pb2.Instruction(fill_vec8_args=foo_pb2.OpFillVec8Args(
        dst_data=ref(TEMP2), scalar_ref=ref(STDDEV)
    )),
    # Loop
    foo_pb2.Instruction(loop_start_args=foo_pb2.OpLoopStartArgs(count=NUM_CHUNKS)),
    foo_pb2.Instruction(block_args=foo_pb2.OpBlockArgs(sequence=phase4_loop_body_seq)),
    foo_pb2.Instruction(loop_end_args=foo_pb2.OpLoopEndArgs()),
    # Return from Phase 4 block
    foo_pb2.Instruction(return_args=foo_pb2.OpReturnArgs())
])

# ========================================================================
# Main Sequence: Execute all phases
# ========================================================================
main_sequence = foo_pb2.InstructionSequence(instructions=[
    foo_pb2.Instruction(block_args=foo_pb2.OpBlockArgs(sequence=phase1_seq)),
    foo_pb2.Instruction(block_args=foo_pb2.OpBlockArgs(sequence=phase2_seq)),
    foo_pb2.Instruction(block_args=foo_pb2.OpBlockArgs(sequence=phase3_seq)),
    foo_pb2.Instruction(block_args=foo_pb2.OpBlockArgs(sequence=phase4_seq)),
    foo_pb2.Instruction(halt_args=foo_pb2.OpHaltArgs())
])

# --- Output the generated sequence ---
# print("Generated Protobuf Instruction Sequence (Text Format):")
# print(text_format.MessageToString(main_sequence))

# Write binary format to file foo.msg
output_filename = "foo.msg"
try:
    with open(output_filename, "wb") as f:
        f.write(main_sequence.SerializeToString())
    print(f"\nSuccessfully generated and wrote z-score instruction sequence to {output_filename}", file=sys.stderr)
except IOError as e:
    print(f"\nError writing to file {output_filename}: {e}", file=sys.stderr)
    sys.exit(1)

# For demonstration, print text format (optional, can be commented out)
# print(text_format.MessageToString(main_sequence))
