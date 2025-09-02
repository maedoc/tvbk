#include "uops.h"
#include <stdio.h>
#include <stdlib.h>

// --- Test Operations ---

// Operation to add an integer value to the argument (int*)
err_t add_int_op(void* arg) {
    if (!arg) return ERR;
    int* value = (int*)arg;
    (*value) += 5;
    printf("add_int_op: value is now %d\n", *value);
    return OK;
}

// Operation to multiply a float value by a constant (float*)
err_t multiply_float_op(void* arg) {
    if (!arg) return ERR;
    float* value = (float*)arg;
    (*value) *= 2.0f;
    printf("multiply_float_op: value is now %f\n", *value);
    return OK;
}

// Operation to print a string (char*) - demonstrates different arg type
err_t print_string_op(void* arg) {
    if (!arg) return ERR;
    char* str = (char*)arg;
    printf("print_string_op: %s\n", str);
    return OK;
}

// --- Main Test Function ---

int main() {
    int int_val = 10;
    float float_val = 3.14f;
    char* string_val = "Hello Sequence!";

    printf("Initial int_val: %d\n", int_val);
    printf("Initial float_val: %f\n", float_val);
    printf("Initial string_val: %s\n", string_val);
    printf("Running sequence...\n");

    // Define the sequence of operations
    op_t ops_to_run[] = {
        add_int_op,
        multiply_float_op,
        print_string_op,
        add_int_op // Run add again
    };

    // Define the arguments for each operation
    void* args_for_ops[] = {
        &int_val,
        &float_val,
        string_val,
        &int_val
    };

    size_t num_ops = sizeof(ops_to_run) / sizeof(ops_to_run[0]);

    // Create the seq_t structure
    seq_t sequence = {
        .ops = ops_to_run,
        .args = args_for_ops,
        .n = num_ops
    };

    // Execute the sequence
    err_t result = seq(sequence);

    if (result != OK) {
        fprintf(stderr, "Sequence execution failed!\n");
        return 1; // Indicate failure
    }

    printf("Sequence finished.\n");
    printf("Final int_val: %d\n", int_val);
    printf("Final float_val: %f\n", float_val);

    // --- Verification ---
    int expected_int = 10 + 5 + 5;
    float expected_float = 3.14f * 2.0f;
    int success = 1;

    if (int_val != expected_int) {
        fprintf(stderr, "Error: Final int_val (%d) does not match expected value (%d)\n", int_val, expected_int);
        success = 0;
    }
     // Use a small tolerance for float comparison
    if (fabsf(float_val - expected_float) > 1e-6) {
       fprintf(stderr, "Error: Final float_val (%f) does not match expected value (%f)\n", float_val, expected_float);
       success = 0;
    }

    if (success) {
        printf("Verification successful!\n");
        return 0; // Indicate success
    } else {
        printf("Verification failed.\n");
        return 1; // Indicate failure
    }
}
