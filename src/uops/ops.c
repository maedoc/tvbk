#include "uops.h"
#include <math.h> // For sinf
#include "loop.c" // Include loop_inline implementation directly

UOPS_API err_t sinf_op_c(void* arg) {
    if (!arg) return ERR;
    float* val_ptr = (float*)arg;
    *val_ptr = sinf(*val_ptr);
    return OK;
}

UOPS_API err_t sinf_vector_loop_composed_c(sinf_vector_loop_args_t* args) {
    if (!args || !args->data || args->len == 0) return ERR;

    // Arguments for the inner vector operation (sinf_vector_op_c)
    vector_op_args_t inner_op_args = {
        .data = args->data,
        .len = args->len
    };

    // Dummy counter required by loop_inline
    int32_t dummy_counter = 0;

    // Arguments for the outer loop (loop_inline)
    loop_t outer_loop_args = {
        .op = sinf_vector_op_c, // The function to call repeatedly
        .arg = &inner_op_args,   // Argument for sinf_vector_op_c (needs pointer)
        .inc = &dummy_counter,   // Pointer to the dummy counter
        .n = args->iterations    // Number of times to call sinf_vector_op_c
    };

    // Call loop_inline to execute the composition
    return loop_inline(outer_loop_args);
}

UOPS_API err_t sinf_loop_c(sinf_loop_args_t* args) {
    if (!args || !args->val_ptr) return ERR;

    float* val_ptr = args->val_ptr;
    size_t n = args->n;

    for (size_t i = 0; i < n; ++i) {
        *val_ptr = sinf(*val_ptr);
    }
    return OK;
}


// --- Vector Ops Implementation ---

UOPS_API err_t sinf_vector_op_c(void* arg) {
    if (!arg) return ERR;
    vector_op_args_t* vec_args = (vector_op_args_t*)arg;
    if (!vec_args->data || vec_args->len == 0) return ERR;

    float* data = vec_args->data;
    size_t len = vec_args->len;

    for (size_t i = 0; i < len; ++i) {
        data[i] = sinf(data[i]);
    }
    return OK;
}

UOPS_API err_t sinf_vector_loop_c(sinf_vector_loop_args_t* args) {
    if (!args || !args->data || args->len == 0) return ERR;

    float* data = args->data;
    size_t len = args->len;
    size_t iterations = args->iterations;

    for (size_t iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < len; ++i) {
            data[i] = sinf(data[i]);
        }
    }
    return OK;
}
