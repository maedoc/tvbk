#include "uops.h"
#include <stdio.h>
#include <assert.h>
#include <math.h> // For powf and fabsf

// Test operation: multiplies the float pointed to by arg by 1.1
err_t multiply_float_op(void* arg) {
    if (!arg) return ERR;
    float* value = (float*)arg;
    (*value) *= 1.1f;
    return OK;
}

int main() {
    int32_t loop_counter = 0; // This will be incremented by loop_api itself
    float op_value = 1.0f;    // This will be modified by our test op
    size_t num_loops = 10;
    err_t result;

    printf("Initial loop_counter: %d\n", loop_counter);
    printf("Initial op_value: %f\n", op_value);
    printf("Looping %zu times...\n", num_loops);

    loop_t args = {
        .op = multiply_float_op,
        .arg = &op_value, // Pass the address of op_value to the op
        .inc = &loop_counter,
        .n = num_loops
    };

    result = loop(args); // Use renamed function

    printf("Final loop_counter: %d\n", loop_counter);
    printf("Final op_value: %f\n", op_value);
    printf("loop result: %s\n", result == OK ? "OK" : "ERR"); // Updated print message

    // Calculate expected value
    float expected_op_value = 1.0f * powf(1.1f, (float)num_loops);
    float tolerance = 1e-6f; // Tolerance for floating point comparison

    // Assertions to check correctness
    assert(result == OK);
    assert(loop_counter == (int32_t)num_loops); // loop_api should increment this n times
    assert(fabsf(op_value - expected_op_value) < tolerance); // Check if op_value matches expected value within tolerance

    printf("Loop test passed!\n");
    return 0; // Success
}
