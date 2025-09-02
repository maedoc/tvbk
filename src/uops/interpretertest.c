#include "interpreter.h"
#include "uops.h" // For err_t, OK, ERR, M_PI_F
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h> // For sinf, fabsf
#include <string.h> // For memcmp, strcmp

#define FLOAT_TOLERANCE 1e-6f

// --- Test Helper Ops (similar to seqtest.c) ---

// Operation to add an integer value to the argument (int*)
err_t test_add_int_op(void* arg) {
    if (!arg) return ERR;
    int* value = (int*)arg;
    (*value) += 5;
    // printf("test_add_int_op: value is now %d\n", *value); // Optional debug print
    return OK;
}

// Operation to multiply a float value by a constant (float*)
err_t test_multiply_float_op(void* arg) {
    if (!arg) return ERR;
    float* value = (float*)arg;
    (*value) *= 2.0f;
    // printf("test_multiply_float_op: value is now %f\n", *value); // Optional debug print
    return OK;
}

// Operation that always fails
err_t test_fail_op(void* arg) {
    // printf("test_fail_op: Intentionally failing.\n"); // Optional debug print
    (void)arg; // Unused
    return ERR;
}


// Helper function to compare floats
int floats_equal(float a, float b) {
    return fabsf(a - b) < FLOAT_TOLERANCE;
}

// --- Test Cases ---

void test_halt() {
    printf("Running test: test_halt\n");
    instruction_t instructions[] = {
        {.tag = OP_HALT}
    };
    err_t result = run_interpreter(instructions, 1);
    assert(result == OK);
    printf("test_halt: OK\n");
}

void test_noop() {
    printf("Running test: test_noop\n");
    instruction_t instructions[] = {
        {.tag = OP_NOOP},
        {.tag = OP_HALT}
    };
    err_t result = run_interpreter(instructions, 2);
    assert(result == OK);
    printf("test_noop: OK\n");
}

void test_sinf_scalar() {
    printf("Running test: test_sinf_scalar\n");
    float value = M_PI_F / 2.0f; // 90 degrees
    float expected = sinf(value);
    instruction_t instructions[] = {
        {.tag = OP_SINF_SCALAR, .args.sinf_scalar = {.val_ptr = &value}},
        {.tag = OP_HALT}
    };
    err_t result = run_interpreter(instructions, 2);
    assert(result == OK);
    assert(floats_equal(value, expected));
    printf("test_sinf_scalar: OK (value=%f, expected=%f)\n", value, expected);
}

void test_sinf_vector() {
    printf("Running test: test_sinf_vector\n");
    float data[] = {0.0f, M_PI_F / 6.0f, M_PI_F / 2.0f}; // 0, 30, 90 degrees
    float expected[] = {sinf(0.0f), sinf(M_PI_F / 6.0f), sinf(M_PI_F / 2.0f)};
    size_t len = sizeof(data) / sizeof(data[0]);

    instruction_t instructions[] = {
        {.tag = OP_SINF_VECTOR, .args.sinf_vector = {.data = data, .len = len}},
        {.tag = OP_HALT}
    };
    err_t result = run_interpreter(instructions, 2);
    assert(result == OK);
    for (size_t i = 0; i < len; ++i) {
        assert(floats_equal(data[i], expected[i]));
    }
    printf("test_sinf_vector: OK\n");
}

void test_loop_simple() {
    printf("Running test: test_loop_simple\n");
    float value = 0.0f; // Value to be modified in the loop
    size_t loop_count = 5;
    float expected = sinf(sinf(sinf(sinf(sinf(0.0f))))); // Apply sinf 5 times

    instruction_t instructions[] = {
        {.tag = OP_LOOP_START, .args.loop_start = {.count = loop_count}}, // ip=0
        {.tag = OP_SINF_SCALAR, .args.sinf_scalar = {.val_ptr = &value}}, // ip=1 (loop body)
        {.tag = OP_LOOP_END},                                            // ip=2
        {.tag = OP_HALT}                                                 // ip=3
    };
    err_t result = run_interpreter(instructions, 4);
    assert(result == OK);
    assert(floats_equal(value, expected));
    printf("test_loop_simple: OK (value=%f, expected=%f)\n", value, expected);
}

void test_loop_zero_iterations() {
    printf("Running test: test_loop_zero_iterations\n");
    float value = 1.23f; // Should remain unchanged
    float original_value = value;

    instruction_t instructions[] = {
        {.tag = OP_LOOP_START, .args.loop_start = {.count = 0}},         // ip=0
        {.tag = OP_SINF_SCALAR, .args.sinf_scalar = {.val_ptr = &value}}, // ip=1 (should be skipped)
        {.tag = OP_LOOP_END},                                            // ip=2
        {.tag = OP_HALT}                                                 // ip=3
    };
    err_t result = run_interpreter(instructions, 4);
    assert(result == OK);
    assert(floats_equal(value, original_value)); // Value should not change
    printf("test_loop_zero_iterations: OK (value=%f)\n", value);
}

void test_loop_nested() {
    printf("Running test: test_loop_nested\n");
    float value = 0.0f;
    size_t outer_loop_count = 2;
    size_t inner_loop_count = 3;
    // Expected: sinf applied 2 * 3 = 6 times
    float expected = sinf(sinf(sinf(sinf(sinf(sinf(0.0f))))));

    instruction_t instructions[] = {
        {.tag = OP_LOOP_START, .args.loop_start = {.count = outer_loop_count}}, // ip=0, Outer loop
        {.tag = OP_LOOP_START, .args.loop_start = {.count = inner_loop_count}}, // ip=1, Inner loop
        {.tag = OP_SINF_SCALAR, .args.sinf_scalar = {.val_ptr = &value}},       // ip=2, Inner loop body
        {.tag = OP_LOOP_END},                                                  // ip=3, End inner loop
        {.tag = OP_LOOP_END},                                                  // ip=4, End outer loop
        {.tag = OP_HALT}                                                       // ip=5
    };
    err_t result = run_interpreter(instructions, 6);
    assert(result == OK);
    assert(floats_equal(value, expected));
    printf("test_loop_nested: OK (value=%f, expected=%f)\n", value, expected);
}

void test_error_no_halt() {
    printf("Running test: test_error_no_halt\n");
    instruction_t instructions[] = {
        {.tag = OP_NOOP}
        // Missing HALT
    };
    err_t result = run_interpreter(instructions, 1);
    assert(result == ERR); // Should fail because it falls off the end
    printf("test_error_no_halt: OK (detected missing HALT)\n");
}

void test_error_loop_end_without_start() {
    printf("Running test: test_error_loop_end_without_start\n");
    instruction_t instructions[] = {
        {.tag = OP_LOOP_END}, // Error: No matching LOOP_START
        {.tag = OP_HALT}
    };
    err_t result = run_interpreter(instructions, 2);
    assert(result == ERR); // Should fail due to stack underflow
    printf("test_error_loop_end_without_start: OK (detected stack underflow)\n");
}

// Note: Testing stack overflow requires MAX_INTERP_STACK + 1 nested loops.
// This can be cumbersome to write manually. We'll assume MAX_INTERP_STACK is reasonably small (e.g., 32).
void test_error_loop_stack_overflow() {
    printf("Running test: test_error_loop_stack_overflow\n");
    // Create MAX_INTERP_STACK + 1 nested loops
    size_t num_instructions = (MAX_INTERP_STACK + 1) * 2 + 1; // START/END for each + HALT
    instruction_t* instructions = malloc(num_instructions * sizeof(instruction_t));
    assert(instructions != NULL);

    size_t i = 0;
    // Create MAX_INTERP_STACK + 1 LOOP_STARTs
    for (int j = 0; j < MAX_INTERP_STACK + 1; ++j) {
        instructions[i++] = (instruction_t){.tag = OP_LOOP_START, .args.loop_start = {.count = 1}};
    }
    // Create MAX_INTERP_STACK + 1 LOOP_ENDs
    for (int j = 0; j < MAX_INTERP_STACK + 1; ++j) {
        instructions[i++] = (instruction_t){.tag = OP_LOOP_END};
    }
    instructions[i++] = (instruction_t){.tag = OP_HALT};
    assert(i == num_instructions);

    err_t result = run_interpreter(instructions, num_instructions);
    assert(result == ERR); // Should fail due to stack overflow
    free(instructions);
    printf("test_error_loop_stack_overflow: OK (detected stack overflow)\n");
}

void test_error_sinf_scalar_null() {
    printf("Running test: test_error_sinf_scalar_null\n");
    instruction_t instructions[] = {
        {.tag = OP_SINF_SCALAR, .args.sinf_scalar = {.val_ptr = NULL}}, // NULL pointer
        {.tag = OP_HALT}
    };
    err_t result = run_interpreter(instructions, 2);
    assert(result == ERR);
    printf("test_error_sinf_scalar_null: OK (detected NULL pointer)\n");
}

void test_error_sinf_vector_null_data() {
    printf("Running test: test_error_sinf_vector_null_data\n");
    instruction_t instructions[] = {
        {.tag = OP_SINF_VECTOR, .args.sinf_vector = {.data = NULL, .len = 5}}, // NULL data
        {.tag = OP_HALT}
    };
    err_t result = run_interpreter(instructions, 2);
    assert(result == ERR);
    printf("test_error_sinf_vector_null_data: OK (detected NULL data)\n");
}

void test_error_sinf_vector_zero_len() {
    printf("Running test: test_error_sinf_vector_zero_len\n");
    float data[] = {1.0f}; // Dummy data, won't be accessed
    instruction_t instructions[] = {
        {.tag = OP_SINF_VECTOR, .args.sinf_vector = {.data = data, .len = 0}}, // Zero length
        {.tag = OP_HALT}
    };
    err_t result = run_interpreter(instructions, 2);
    assert(result == ERR);
    printf("test_error_sinf_vector_zero_len: OK (detected zero length)\n");
}

void test_seq_basic() {
    printf("Running test: test_seq_basic\n");
    int int_val = 10;
    float float_val = 3.14f;
    int expected_int = 10 + 5 + 5;
    float expected_float = 3.14f * 2.0f;

    // Define the sequence of operations
    op_t ops_to_run[] = {
        test_add_int_op,
        test_multiply_float_op,
        test_add_int_op // Run add again
    };

    // Define the arguments for each operation
    void* args_for_ops[] = {
        &int_val,
        &float_val,
        &int_val
    };

    size_t num_ops = sizeof(ops_to_run) / sizeof(ops_to_run[0]);

    instruction_t instructions[] = {
        {.tag = OP_SEQ, .args.seq = {.ops = ops_to_run, .args = args_for_ops, .n = num_ops}},
        {.tag = OP_HALT}
    };

    err_t result = run_interpreter(instructions, 2);
    assert(result == OK);
    assert(int_val == expected_int);
    assert(floats_equal(float_val, expected_float));

    printf("test_seq_basic: OK (int=%d, float=%f)\n", int_val, float_val);
}

void test_seq_error_propagation() {
    printf("Running test: test_seq_error_propagation\n");
    int int_val = 10; // Should be modified by the first op
    int expected_int = 10 + 5;

    op_t ops_to_run[] = {
        test_add_int_op,
        test_fail_op, // This one will fail
        test_add_int_op  // This should not run
    };
    void* args_for_ops[] = { &int_val, NULL, &int_val }; // Args for the ops
    size_t num_ops = sizeof(ops_to_run) / sizeof(ops_to_run[0]);

    instruction_t instructions[] = {
        {.tag = OP_SEQ, .args.seq = {.ops = ops_to_run, .args = args_for_ops, .n = num_ops}},
        {.tag = OP_HALT} // Should not be reached
    };

    err_t result = run_interpreter(instructions, 2);
    assert(result == ERR); // Expecting failure
    assert(int_val == expected_int); // Verify the first op ran, but the third didn't

    printf("test_seq_error_propagation: OK (detected failure, int=%d)\n", int_val);
}

// --- Block Tests ---

void test_block_simple() {
    printf("Running test: test_block_simple\n");
    float value = M_PI_F / 6.0f; // 30 degrees
    float expected = sinf(value);

    // Block instructions
    instruction_t block_instructions[] = {
        {.tag = OP_SINF_SCALAR, .args.sinf_scalar = {.val_ptr = &value}},
        {.tag = OP_RETURN}
    };
    size_t block_num = sizeof(block_instructions) / sizeof(block_instructions[0]);

    // Main instructions
    instruction_t main_instructions[] = {
        {.tag = OP_BLOCK, .args.block = {.instructions = block_instructions, .num_instructions = block_num}},
        {.tag = OP_HALT}
    };
    size_t main_num = sizeof(main_instructions) / sizeof(main_instructions[0]);

    err_t result = run_interpreter(main_instructions, main_num);
    assert(result == OK);
    assert(floats_equal(value, expected));
    printf("test_block_simple: OK (value=%f)\n", value);
}

void test_block_in_loop() {
    printf("Running test: test_block_in_loop\n");
    float value = 0.0f;
    size_t loop_count = 4;
    float expected = sinf(sinf(sinf(sinf(0.0f)))); // sinf applied 4 times

    // Block instructions (apply sinf once and return)
    instruction_t block_instructions[] = {
        {.tag = OP_SINF_SCALAR, .args.sinf_scalar = {.val_ptr = &value}},
        {.tag = OP_RETURN}
    };
    size_t block_num = sizeof(block_instructions) / sizeof(block_instructions[0]);

    // Main instructions (loop calling the block)
    instruction_t main_instructions[] = {
        {.tag = OP_LOOP_START, .args.loop_start = {.count = loop_count}}, // ip=0
        {.tag = OP_BLOCK, .args.block = {.instructions = block_instructions, .num_instructions = block_num}}, // ip=1
        {.tag = OP_LOOP_END},                                            // ip=2
        {.tag = OP_HALT}                                                 // ip=3
    };
    size_t main_num = sizeof(main_instructions) / sizeof(main_instructions[0]);

    err_t result = run_interpreter(main_instructions, main_num);
    assert(result == OK);
    assert(floats_equal(value, expected));
    printf("test_block_in_loop: OK (value=%f)\n", value);
}

void test_loop_in_block() {
    printf("Running test: test_loop_in_block\n");
    float value = 0.0f;
    size_t loop_count = 3;
    float expected = sinf(sinf(sinf(0.0f))); // sinf applied 3 times

    // Block instructions (contains the loop)
    instruction_t block_instructions[] = {
        {.tag = OP_LOOP_START, .args.loop_start = {.count = loop_count}}, // ip=0 (block)
        {.tag = OP_SINF_SCALAR, .args.sinf_scalar = {.val_ptr = &value}}, // ip=1 (block)
        {.tag = OP_LOOP_END},                                            // ip=2 (block)
        {.tag = OP_RETURN}                                               // ip=3 (block)
    };
    size_t block_num = sizeof(block_instructions) / sizeof(block_instructions[0]);

    // Main instructions (just calls the block)
    instruction_t main_instructions[] = {
        {.tag = OP_BLOCK, .args.block = {.instructions = block_instructions, .num_instructions = block_num}}, // ip=0
        {.tag = OP_HALT}                                                                                    // ip=1
    };
    size_t main_num = sizeof(main_instructions) / sizeof(main_instructions[0]);

    err_t result = run_interpreter(main_instructions, main_num);
    assert(result == OK);
    assert(floats_equal(value, expected));
    printf("test_loop_in_block: OK (value=%f)\n", value);
}

void test_error_return_without_block() {
    printf("Running test: test_error_return_without_block\n");
    instruction_t instructions[] = {
        {.tag = OP_RETURN}, // Error: No block context
        {.tag = OP_HALT}
    };
    err_t result = run_interpreter(instructions, 2);
    assert(result == ERR);
    printf("test_error_return_without_block: OK (detected stack underflow)\n");
}

void test_error_block_stack_overflow() {
    printf("Running test: test_error_block_stack_overflow\n");
    // Create MAX_INTERP_STACK + 1 nested blocks

    // Innermost block
    instruction_t innermost_block[] = {{.tag = OP_RETURN}};
    size_t innermost_num = 1;

    // Dynamically build the nested structure
    instruction_t* current_block_instrs = innermost_block;
    size_t current_block_num = innermost_num;

    instruction_t* blocks_to_free[MAX_INTERP_STACK + 1];
    int free_idx = 0;

    for (int i = 0; i < MAX_INTERP_STACK + 1; ++i) {
        instruction_t* outer_block_instrs = malloc(2 * sizeof(instruction_t)); // OP_BLOCK + OP_RETURN/HALT
        assert(outer_block_instrs != NULL);
        blocks_to_free[free_idx++] = outer_block_instrs;

        outer_block_instrs[0] = (instruction_t){.tag = OP_BLOCK, .args.block = {.instructions = current_block_instrs, .num_instructions = current_block_num}};
        outer_block_instrs[1] = (instruction_t){.tag = (i == 0) ? OP_HALT : OP_RETURN}; // Top level halts, others return

        current_block_instrs = outer_block_instrs;
        current_block_num = 2;
    }

    // Run the outermost block (which is now current_block_instrs)
    err_t result = run_interpreter(current_block_instrs, current_block_num);
    assert(result == ERR); // Should fail due to block stack overflow

    // Free allocated memory (innermost_block was static)
    for(int i = 0; i < free_idx; ++i) {
        // Important: Only free the instruction array, not the nested ones it points to if they are also in the list
        // Check if the instructions pointer is the same as any previously freed block to avoid double free
        int already_freed = 0;
        for (int j = i + 1; j < free_idx; ++j) {
             if (blocks_to_free[i] == blocks_to_free[j]->args.block.instructions) {
                 already_freed = 1;
                 break;
             }
        }
        // Also check against the static innermost block
        if (blocks_to_free[i]->args.block.instructions == innermost_block) {
             already_freed = 1;
        }

        // A simpler approach for this specific test structure: just free the outer arrays
        // We know we allocated MAX_INTERP_STACK + 1 arrays.
        // Let's just free them directly based on the blocks_to_free array.
    }
     for(int i = 0; i < free_idx; ++i) {
         free(blocks_to_free[i]);
     }


    printf("test_error_block_stack_overflow: OK (detected stack overflow)\n");
}


void test_error_fall_off_block() {
    printf("Running test: test_error_fall_off_block\n");
    // Block instructions (missing OP_RETURN)
    instruction_t block_instructions[] = {
        {.tag = OP_NOOP}
        // Missing OP_RETURN
    };
    size_t block_num = sizeof(block_instructions) / sizeof(block_instructions[0]);

    instruction_t main_instructions[] = {
        {.tag = OP_BLOCK, .args.block = {.instructions = block_instructions, .num_instructions = block_num}},
        {.tag = OP_HALT} // Should not be reached
    };
    size_t main_num = sizeof(main_instructions) / sizeof(main_instructions[0]);

    err_t result = run_interpreter(main_instructions, main_num);
    assert(result == ERR);
    printf("test_error_fall_off_block: OK (detected missing OP_RETURN)\n");
}

void test_error_block_null_instructions() {
     printf("Running test: test_error_block_null_instructions\n");
     instruction_t instructions[] = {
        {.tag = OP_BLOCK, .args.block = {.instructions = NULL, .num_instructions = 5}}, // NULL instructions
        {.tag = OP_HALT}
    };
    err_t result = run_interpreter(instructions, 2);
    assert(result == ERR);
    printf("test_error_block_null_instructions: OK (detected NULL instructions)\n");
}

void test_error_block_zero_num() {
     printf("Running test: test_error_block_zero_num\n");
     instruction_t dummy_instr = {.tag = OP_HALT}; // Need a non-NULL pointer
     instruction_t instructions[] = {
        {.tag = OP_BLOCK, .args.block = {.instructions = &dummy_instr, .num_instructions = 0}}, // Zero num_instructions
        {.tag = OP_HALT}
    };
    err_t result = run_interpreter(instructions, 2);
    assert(result == ERR);
    printf("test_error_block_zero_num: OK (detected zero num_instructions)\n");
}


int main() {
    printf("Starting interpreter tests...\n");

    test_halt();
    test_noop();
    test_sinf_scalar();
    test_sinf_vector();
    test_loop_simple();
    test_loop_zero_iterations();
    test_loop_nested();

    // Error condition tests
    test_error_no_halt();
    test_error_loop_end_without_start();
    test_error_loop_stack_overflow();
    test_error_sinf_scalar_null();
    test_error_sinf_vector_null_data();
    test_error_sinf_vector_zero_len();

    // Seq tests
    test_seq_basic();
    test_seq_error_propagation();

    // Block tests
    test_block_simple();
    test_block_in_loop();
    test_loop_in_block();
    test_error_return_without_block();
    test_error_block_stack_overflow();
    test_error_fall_off_block();
    test_error_block_null_instructions();
    test_error_block_zero_num();


    printf("\nAll interpreter tests passed!\n");
    return 0;
}
