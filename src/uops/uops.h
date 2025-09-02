#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define M_PI 3.14159265358979323846
#define M_PI_F ((float) M_PI)

// Define UOPS_API for DLL export/import
#if defined(_MSC_VER) // Check if the compiler is Microsoft Visual C++
  #if defined(UOPS_BUILD_DLL) // Building the DLL
    #define UOPS_API __declspec(dllexport)
  #elif defined(UOPS_BUILD_STATIC) // Building the static library
    #define UOPS_API // No special declaration needed for static lib definitions
  #else // Using the DLL
    #define UOPS_API __declspec(dllimport)
  #endif
#else // For non-MSVC compilers (GCC, Clang, etc.)
  #define UOPS_API // Define as empty, relying on default visibility or other attributes if needed later
#endif


#ifdef DONTFORCEINLINE
#define INLINE
#else
  #ifdef _MSC_VER
  #include <intrin.h>
  #define INLINE __forceinline
  #else
  #define INLINE __attribute__((always_inline)) inline
  #endif
#endif

// typedefs
typedef struct scan_f { void*c,*x,*y; } scan_f_t;
typedef enum err { OK, ERR } err_t;
typedef err_t(*scan_fn)(scan_f_t);
typedef err_t(*op_t)(void*);

// structs
/**
 * @brief Arguments for the scan operation.
 */
typedef struct {
    const scan_fn f; ///< The scan function to apply element-wise.
    void *c;         ///< Pointer to constant data (optional, passed to f).
    void *x;         ///< Pointer to the input array.
    void *y;         ///< Pointer to the output array.
    size_t sz;       ///< Size (in bytes) of each element in x and y.
    size_t n;        ///< Number of elements in x and y.
} scan_t;

/**
 * @brief Arguments for the loop operation.
 */
typedef struct {
    const op_t op;
    void *arg;    // Argument to pass to the op function
    int32_t *inc; // Pointer to the integer to be incremented
    size_t n;     // Number of times to loop
} loop_t;

typedef struct {
    const op_t* ops; // Array of operations to execute
    void** args;     // Array of arguments corresponding to each op
    size_t n;        // Number of operations in the sequence
} seq_t;

typedef struct {
    const op_t* ops; // Array of operations to execute in parallel
    void** args;     // Array of arguments corresponding to each op
    size_t n;        // Number of operations
    size_t num_threads; // Number of threads to use (0 means auto/default)
} pmap_t;


UOPS_API err_t scan(scan_t s);
UOPS_API err_t loop(loop_t l);
UOPS_API err_t seq(seq_t s);
UOPS_API err_t pmap(pmap_t p);

// ops designed for microbenchmarking, also need export

/**
 * @brief Applies sinf once to the float pointed to by arg. (Scalar Op)
 * @param arg A void pointer, expected to point to a float.
 * @return OK on success, ERR if arg is NULL.
 */
UOPS_API err_t sinf_op_c(void* arg);

/**
 * @brief Arguments for sinf_loop_c (Scalar internal loop).
 */
typedef struct sinf_loop_args {
    float* val_ptr; // Pointer to the float value to modify
    size_t n;       ///< Number of iterations (scalar applies)
} sinf_loop_args_t;

/**
 * @brief Applies sinf N times to the float pointed to by args->val_ptr. (Scalar internal loop)
 * @param args Pointer to a sinf_loop_args_t struct containing the float pointer and iteration count.
 * @return OK on success, ERR if args or args->val_ptr is NULL.
 */
UOPS_API err_t sinf_loop_c(sinf_loop_args_t* args);


// --- Vector Ops ---

/**
 * @brief Arguments for vector operations used as the `arg` for loop_api's op_t.
 */
typedef struct vector_op_args {
    float* data;    ///< Pointer to the vector data
    size_t len;     ///< Number of elements in the vector
} vector_op_args_t;

/**
 * @brief Applies sinf element-wise to the float vector specified in arg. (Vector Op for loop_api)
 * @param arg A void pointer, expected to point to a vector_op_args_t struct.
 * @return OK on success, ERR if arg, arg->data is NULL or len is 0.
 */
UOPS_API err_t sinf_vector_op_c(void* arg);

/**
 * @brief Arguments for sinf_vector_loop_c and sinf_vector_loop_composed_c.
 */
typedef struct sinf_vector_loop_args {
    float* data;    ///< Pointer to the vector data
    size_t len;     ///< Number of elements in the vector
    size_t iterations; ///< Number of times to apply the vector op
} sinf_vector_loop_args_t;

/**
 * @brief Applies sinf element-wise to a float vector, repeating the process 'iterations' times. (Handwritten internal loop)
 * @param args Pointer to a sinf_vector_loop_args_t struct.
 * @return OK on success, ERR if args, args->data is NULL or len is 0.
 */
UOPS_API err_t sinf_vector_loop_c(sinf_vector_loop_args_t* args);

/**
 * @brief Applies sinf element-wise to a float vector, repeating the process 'iterations' times,
 *        by composing loop_inline and sinf_vector_op_c within C. (Composed internal loop)
 * @param args Pointer to a sinf_vector_loop_args_t struct.
 * @return OK on success, ERR if args, args->data is NULL or len is 0, or if internal loop fails.
 */
UOPS_API err_t sinf_vector_loop_composed_c(sinf_vector_loop_args_t* args);

