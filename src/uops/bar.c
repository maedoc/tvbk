#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h> // For malloc, free
#include <stdio.h>  // For error messages
#include <string.h> // For memcpy
#include <omp.h>    // Include OpenMP header
#include <time.h>   // For clock_gettime

#if defined(__GNUC__) || defined(__clang__)
#define GCC_ALIGNED_64 __attribute__((aligned(64)))
#else
#define GCC_ALIGNED_64
#endif

#define VECTOR_SIZE 32
#define MAX_INTERP_STACK 32

// Forward declaration for instruction_t, used in zscore_program_t
typedef struct instruction_s instruction_t;

// Struct to hold all allocated instruction arrays for the z-score program
typedef struct {
    instruction_t* main_sequence;
    instruction_t* phase1_loop_body;
    instruction_t* phase1_instructions;
    instruction_t* phase2_instructions;
    instruction_t* phase3_instructions;
    instruction_t* phase4_loop_body;
    instruction_t* phase4_instructions;
} zscore_program_t;

// Forward declarations for z-score program functions
typedef enum { OK, ERR } err_t; // Moved err_t definition here

zscore_program_t build_zscore_instructions(size_t N);
err_t compute_zscore_compiled(const float* x_in, float* z_out, const zscore_program_t* program);
void free_zscore_instructions(zscore_program_t* program);

typedef enum {
    OP_HALT,
    OP_LOOP_START,
    OP_LOOP_END,
    OP_BLOCK,
    OP_RETURN,
    OP_ADD_VEC8,
    OP_MUL_VEC8,
    OP_EXP_VEC8,
    OP_SIN_VEC8,
    OP_COS_VEC8,
    OP_SUB_VEC8,    // New
    OP_DIV_VEC8,    // New
    OP_POW_VEC8,    // New
    OP_SHEUNPRED_VEC8, // New: xi = x + dt*dx + z
    OP_SHEUNCORR_VEC8, // New: nx = x + dt/2*(dx1+dx2) + z
    OP_DOT_VEC8,    // New: result = sum(x[i]*y[i])
    OP_ZERO_VEC8,   // New: Zero out a vector
    OP_SUM_VEC8,    // New: Sum elements of a vector to a scalar pointer
    OP_SQRT_VEC8,   // New: Element-wise square root
    OP_FILL_VEC8,   // New: Fill vector with a scalar value
    OP_COPY_VEC8,   // New: Copy vector src1 to dst
    OP_PINC,        // New: Increment a pointer by N bytes
    OP_MEAN_VEC8,   // New: Mean elements of a vector to a scalar pointer
    OP_DIV_SCALAR,  // New: Divide scalar *dst_ptr by *src_ptr, store in *dst_ptr
    OP_SUB_SCALAR,  // New: Subtract scalar *src_ptr from *dst_ptr, store in *dst_ptr
    OP_MUL_SCALAR,  // New: Multiply scalar *dst_ptr by *src_ptr, store in *dst_ptr
    OP_SQRT_SCALAR, // New: Compute sqrt(*src_ptr), store in *dst_ptr
    OP_COPY_SCALAR, // New: Copy scalar *src_ptr to *dst_ptr
    OP_DATA_VEC8    // New: Defines an embedded data vector
} op_type_t;

typedef struct {
    float* dst_data;  // Destination operand
    float* src1_data; // First source operand
    float* src2_data; // Second source operand (for binary ops like ADD, MUL, SUB, DIV, POW)
} op_args_vec8_t;

typedef struct {
    float* xi_data; // Output: xi
    float* x_data;  // Input: x
    float* dt_ptr;  // Input: dt (pointer to scalar)
    float* dx_data; // Input: dx
    float* z_data;  // Input: z
} op_args_sheunpred_t;

typedef struct {
    float* nx_data; // Output: nx
    float* x_data;  // Input: x
    float* dt_ptr;  // Input: dt (pointer to scalar)
    float* dx1_data;// Input: dx1
    float* dx2_data;// Input: dx2
    float* z_data;  // Input: z
} op_args_sheuncorr_t;

typedef struct {
    float* x_data;      // Input vector 1
    float* y_data;      // Input vector 2
    float* result_ptr;  // Output scalar result pointer
} op_args_dot_vec8_t;

typedef struct {
    float* src1_data;   // Input vector
    float* result_ptr;  // Output scalar result pointer
} op_args_sum_vec8_t;

typedef struct {
    float* dst_data;    // Destination vector
    float* scalar_ptr;  // Pointer to the scalar value to fill with
} op_args_fill_vec8_t;

typedef struct {
    void** p;   // Pointer to the pointer to increment
    int    inc; // Increment in bytes (can be negative)
} op_args_pinc_t;

typedef struct {
    float* dst_ptr; // Pointer to destination scalar (also first operand)
    float* src_ptr; // Pointer to source scalar (second operand)
} op_args_scalar_binary_t; // Used for DIV, SUB, MUL (in-place on dst_ptr)

typedef struct {
    float* dst_ptr; // Pointer to destination scalar
    float* src_ptr; // Pointer to source scalar
} op_args_scalar_unary_t; // Used for SQRT, COPY

typedef struct {
    float data[VECTOR_SIZE]; // Embedded data vector
} op_args_data_vec8_t;

typedef struct {
    size_t count;
} op_args_loop_start_t;

// typedef struct instruction_s instruction_t; // Moved earlier as forward declaration

typedef struct {
    instruction_t* instructions;
    // size_t         num_instructions; // Removed
} op_args_block_t;

typedef union {
    op_args_vec8_t       vec8_op;    // Used by ADD, MUL, SUB, DIV, POW, EXP, SIN, COS
    op_args_loop_start_t loop_start;
    op_args_block_t      block;
    op_args_sheunpred_t  sheunpred;
    op_args_sheuncorr_t  sheuncorr;
    op_args_dot_vec8_t   dot_vec8;
    op_args_sum_vec8_t   sum_vec8;   // Also used by OP_MEAN_VEC8
    op_args_fill_vec8_t  fill_vec8;
    op_args_pinc_t       pinc;
    op_args_scalar_binary_t scalar_binary; // Used by DIV, SUB, MUL
    op_args_scalar_unary_t  scalar_unary;  // Used by SQRT, COPY
    op_args_data_vec8_t  data_vec8;    // Used by OP_DATA_VEC8
    // ZERO, SQRT_VEC8, COPY_VEC8 reuse vec8_op args (dst_data, src1_data)
} op_args_union_t;

struct instruction_s {
    op_type_t       tag;
    op_args_union_t args;
};

typedef struct {
    size_t loop_counter;
    size_t loop_start_ip;
} loop_stack_entry_t;

typedef struct {
    instruction_t* parent_instructions;
    // size_t         parent_num_instructions; // Removed
    size_t         return_ip;
} interp_state_stack_entry_t;

// Assumes loops are correctly nested and terminated
static size_t find_matching_loop_end(const instruction_t* instructions, size_t start_ip) {
    size_t nesting_level = 1;
    size_t current_ip = start_ip;
    // Loop indefinitely until the matching end is found
    while (1) {
        op_type_t tag = instructions[current_ip].tag;
        // Add basic check for HALT/RETURN as safety? No, rely on validation.
        if (tag == OP_LOOP_START) nesting_level++;
        else if (tag == OP_LOOP_END) {
            nesting_level--;
            if (nesting_level == 0) return current_ip + 1;
        }
        current_ip++;
    }
    return (size_t)-1; // Should not be reached if loops are well-formed
}

err_t run_interpreter(instruction_t* instructions) {
    size_t ip = 0;
    loop_stack_entry_t loop_stack[MAX_INTERP_STACK];
    int loop_sp = -1;
    interp_state_stack_entry_t interp_state_stack[MAX_INTERP_STACK];
    int interp_sp = -1;

    instruction_t* current_instructions = instructions;
    // size_t current_num_instructions = num_instructions; // Removed

    // Dispatch table for computed gotos (GCC/Clang extension)
    static void* dispatch_table[] = {
        &&HANDLE_OP_HALT, &&HANDLE_OP_LOOP_START, &&HANDLE_OP_LOOP_END,
        &&HANDLE_OP_BLOCK, &&HANDLE_OP_RETURN, &&HANDLE_OP_ADD_VEC8,
        &&HANDLE_OP_MUL_VEC8, &&HANDLE_OP_EXP_VEC8, &&HANDLE_OP_SIN_VEC8,
        &&HANDLE_OP_COS_VEC8, &&HANDLE_OP_SUB_VEC8, &&HANDLE_OP_DIV_VEC8,
        &&HANDLE_OP_POW_VEC8, &&HANDLE_OP_SHEUNPRED_VEC8, &&HANDLE_OP_SHEUNCORR_VEC8,
        &&HANDLE_OP_DOT_VEC8, &&HANDLE_OP_ZERO_VEC8, &&HANDLE_OP_SUM_VEC8,
        &&HANDLE_OP_SQRT_VEC8, &&HANDLE_OP_FILL_VEC8, &&HANDLE_OP_COPY_VEC8,
        &&HANDLE_OP_PINC, &&HANDLE_OP_MEAN_VEC8, &&HANDLE_OP_DIV_SCALAR,
        &&HANDLE_OP_SUB_SCALAR, &&HANDLE_OP_MUL_SCALAR, &&HANDLE_OP_SQRT_SCALAR,
        &&HANDLE_OP_COPY_SCALAR, &&HANDLE_OP_DATA_VEC8
    };
    // Check if the number of entries matches the number of opcodes
    // This is a compile-time check if possible, or a runtime check otherwise.
    // For simplicity, we assume the table is correct. A production system might
    // add checks or generate this table.
    #define NUM_OPCODES (sizeof(dispatch_table) / sizeof(dispatch_table[0]))

    // Macro for dispatching to the next instruction handler
    #define DISPATCH() \
        do { \
            current_instruction = &current_instructions[ip]; \
            op_type_t tag = current_instruction->tag; \
            if (tag >= NUM_OPCODES) goto HANDLE_INVALID_OPCODE; \
            goto *dispatch_table[tag]; \
        } while (0)

    // Start execution by dispatching the first instruction
    const instruction_t* current_instruction; // Declare here for use in DISPATCH and handlers
    DISPATCH();

    // --- Opcode Handlers ---

    HANDLE_OP_HALT:
        return OK;

    HANDLE_OP_LOOP_START: {
        size_t count = current_instruction->args.loop_start.count;
        if (count == 0) {
            ip = find_matching_loop_end(current_instructions, ip + 1);
            DISPATCH(); // Jump to instruction after loop end
        } else {
            loop_sp++;
            // TODO: Add stack overflow check
            loop_stack[loop_sp].loop_counter = count;
            loop_stack[loop_sp].loop_start_ip = ip + 1;
            ip++;
            DISPATCH(); // Continue to next instruction (loop body)
        }
    }

    HANDLE_OP_LOOP_END: {
        if (loop_sp < 0) return ERR; // Unmatched loop end
        if (loop_stack[loop_sp].loop_counter > 0) {
             loop_stack[loop_sp].loop_counter--;
        }
        if (loop_stack[loop_sp].loop_counter > 0) {
            ip = loop_stack[loop_sp].loop_start_ip;
            DISPATCH(); // Jump back to loop start
        } else {
            loop_sp--;
            ip++;
            DISPATCH(); // Continue after loop end
        }
    }

    HANDLE_OP_BLOCK: {
        const op_args_block_t* block_args = &current_instruction->args.block;
        interp_sp++;
        // TODO: Add stack overflow check
        interp_state_stack[interp_sp].parent_instructions = current_instructions;
        interp_state_stack[interp_sp].return_ip = ip + 1;
        current_instructions = block_args->instructions;
        ip = 0;
        DISPATCH(); // Jump to first instruction of the block
    }

    HANDLE_OP_RETURN: {
        if (interp_sp < 0) return ERR; // Return from base level
        current_instructions = interp_state_stack[interp_sp].parent_instructions;
        ip = interp_state_stack[interp_sp].return_ip;
        interp_sp--;
        DISPATCH(); // Jump to instruction after the block call
    }

    HANDLE_OP_ADD_VEC8: {
        float* GCC_ALIGNED_64 dst = current_instruction->args.vec8_op.dst_data;
        float* GCC_ALIGNED_64 src1 = current_instruction->args.vec8_op.src1_data;
        float* GCC_ALIGNED_64 src2 = current_instruction->args.vec8_op.src2_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = src1[i] + src2[i];
        ip++;
        DISPATCH();
    }

    HANDLE_OP_MUL_VEC8: {
        float* GCC_ALIGNED_64 dst = current_instruction->args.vec8_op.dst_data;
        float* GCC_ALIGNED_64 src1 = current_instruction->args.vec8_op.src1_data;
        float* GCC_ALIGNED_64 src2 = current_instruction->args.vec8_op.src2_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = src1[i] * src2[i];
        ip++;
        DISPATCH();
    }

     HANDLE_OP_EXP_VEC8: {
        float* GCC_ALIGNED_64 dst = current_instruction->args.vec8_op.dst_data;
        float* GCC_ALIGNED_64 src1 = current_instruction->args.vec8_op.src1_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = expf(src1[i]);
        ip++;
        DISPATCH();
    }

    HANDLE_OP_SIN_VEC8: {
        float* GCC_ALIGNED_64 dst = current_instruction->args.vec8_op.dst_data;
        float* GCC_ALIGNED_64 src1 = current_instruction->args.vec8_op.src1_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = sinf(src1[i]);
        ip++;
        DISPATCH();
    }

    HANDLE_OP_COS_VEC8: {
        float* GCC_ALIGNED_64 dst = current_instruction->args.vec8_op.dst_data;
        float* GCC_ALIGNED_64 src1 = current_instruction->args.vec8_op.src1_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = cosf(src1[i]);
        ip++;
        DISPATCH();
    }

    HANDLE_OP_SUB_VEC8: {
        float* GCC_ALIGNED_64 dst = current_instruction->args.vec8_op.dst_data;
        float* GCC_ALIGNED_64 src1 = current_instruction->args.vec8_op.src1_data;
        float* GCC_ALIGNED_64 src2 = current_instruction->args.vec8_op.src2_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = src1[i] - src2[i];
        ip++;
        DISPATCH();
    }

    HANDLE_OP_DIV_VEC8: {
        float* GCC_ALIGNED_64 dst = current_instruction->args.vec8_op.dst_data;
        float* GCC_ALIGNED_64 src1 = current_instruction->args.vec8_op.src1_data;
        float* GCC_ALIGNED_64 src2 = current_instruction->args.vec8_op.src2_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = src1[i] / src2[i];
        ip++;
        DISPATCH();
    }

     HANDLE_OP_POW_VEC8: {
        float* GCC_ALIGNED_64 dst = current_instruction->args.vec8_op.dst_data;
        float* GCC_ALIGNED_64 src1 = current_instruction->args.vec8_op.src1_data;
        float* GCC_ALIGNED_64 src2 = current_instruction->args.vec8_op.src2_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = powf(src1[i], src2[i]);
        ip++;
        DISPATCH();
    }

    HANDLE_OP_SHEUNPRED_VEC8: {
        const op_args_sheunpred_t* args = &current_instruction->args.sheunpred;
        float* GCC_ALIGNED_64 xi = args->xi_data;
        float* GCC_ALIGNED_64 x = args->x_data;
        float dt = *(args->dt_ptr);
        float* GCC_ALIGNED_64 dx = args->dx_data;
        float* GCC_ALIGNED_64 z = args->z_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            xi[i] = x[i] + dt * dx[i] + z[i];
        }
        ip++;
        DISPATCH();
    }

    HANDLE_OP_SHEUNCORR_VEC8: {
        const op_args_sheuncorr_t* args = &current_instruction->args.sheuncorr;
        float* GCC_ALIGNED_64 nx = args->nx_data;
        float* GCC_ALIGNED_64 x = args->x_data;
        float dt = *(args->dt_ptr);
        float* GCC_ALIGNED_64 dx1 = args->dx1_data;
        float* GCC_ALIGNED_64 dx2 = args->dx2_data;
        float* GCC_ALIGNED_64 z = args->z_data;
        float dt_half = dt / 2.0f;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            nx[i] = x[i] + dt_half * (dx1[i] + dx2[i]) + z[i];
        }
        ip++;
        DISPATCH();
    }

    HANDLE_OP_DOT_VEC8: {
        const op_args_dot_vec8_t* args = &current_instruction->args.dot_vec8;
        float* GCC_ALIGNED_64 x = args->x_data;
        float* GCC_ALIGNED_64 y = args->y_data;
        float* result_ptr = args->result_ptr;
        float p = 0.0f;
        #pragma omp simd reduction(+:p)
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            p += x[i] * y[i];
        }
        *result_ptr = p;
        ip++;
        DISPATCH();
    }

    HANDLE_OP_ZERO_VEC8: {
        float* GCC_ALIGNED_64 dst = current_instruction->args.vec8_op.dst_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = 0.0f;
        ip++;
        DISPATCH();
    }

    HANDLE_OP_SUM_VEC8: {
        const op_args_sum_vec8_t* args = &current_instruction->args.sum_vec8;
        float* GCC_ALIGNED_64 src = args->src1_data;
        float* result_ptr = args->result_ptr;
        float p = 0.0f;
        #pragma omp simd reduction(+:p)
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            p += src[i];
        }
        *result_ptr = p;
        ip++;
        DISPATCH();
    }

    HANDLE_OP_MEAN_VEC8: {
        const op_args_sum_vec8_t* args = &current_instruction->args.sum_vec8;
        float* GCC_ALIGNED_64 src = args->src1_data;
        float* result_ptr = args->result_ptr;
        float p = 0.0f;
        #pragma omp simd reduction(+:p)
        for (int i = 0; i < VECTOR_SIZE; ++i) {
            p += src[i];
        }
        *result_ptr = p / (float)VECTOR_SIZE;
        ip++;
        DISPATCH();
    }

    HANDLE_OP_SQRT_VEC8: {
        float* GCC_ALIGNED_64 dst = current_instruction->args.vec8_op.dst_data;
        float* GCC_ALIGNED_64 src1 = current_instruction->args.vec8_op.src1_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = sqrtf(src1[i]);
        ip++;
        DISPATCH();
    }

    HANDLE_OP_FILL_VEC8: {
        const op_args_fill_vec8_t* args = &current_instruction->args.fill_vec8;
        float* GCC_ALIGNED_64 dst = args->dst_data;
        float scalar_val = *(args->scalar_ptr);
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = scalar_val;
        ip++;
        DISPATCH();
    }

     HANDLE_OP_COPY_VEC8: {
        float* GCC_ALIGNED_64 dst = current_instruction->args.vec8_op.dst_data;
        float* GCC_ALIGNED_64 src1 = current_instruction->args.vec8_op.src1_data;
        #pragma omp simd
        for (int i = 0; i < VECTOR_SIZE; ++i) dst[i] = src1[i];
        ip++;
        DISPATCH();
    }

    HANDLE_OP_PINC: {
        const op_args_pinc_t* args = &current_instruction->args.pinc;
        if (!args->p) return ERR;
        *args->p = (void*)((char*)(*args->p) + args->inc);
        ip++;
        DISPATCH();
    }

    HANDLE_OP_DIV_SCALAR: {
        const op_args_scalar_binary_t* args = &current_instruction->args.scalar_binary;
        *args->dst_ptr = *args->dst_ptr / *args->src_ptr;
        ip++;
        DISPATCH();
    }

    HANDLE_OP_SUB_SCALAR: {
        const op_args_scalar_binary_t* args = &current_instruction->args.scalar_binary;
        *args->dst_ptr = *args->dst_ptr - *args->src_ptr;
        ip++;
        DISPATCH();
    }

    HANDLE_OP_MUL_SCALAR: {
        const op_args_scalar_binary_t* args = &current_instruction->args.scalar_binary;
        *args->dst_ptr = *args->dst_ptr * *args->src_ptr;
        ip++;
        DISPATCH();
    }

    HANDLE_OP_SQRT_SCALAR: {
        const op_args_scalar_unary_t* args = &current_instruction->args.scalar_unary;
        *args->dst_ptr = sqrtf(*args->src_ptr);
        ip++;
        DISPATCH();
    }

    HANDLE_OP_COPY_SCALAR: {
        const op_args_scalar_unary_t* args = &current_instruction->args.scalar_unary;
        *args->dst_ptr = *args->src_ptr;
        ip++;
        DISPATCH();
    }

    HANDLE_OP_DATA_VEC8: {
        // No-op during execution
        ip++;
        DISPATCH();
    }

    HANDLE_INVALID_OPCODE:
        fprintf(stderr, "[interp] error: unknown or invalid opcode %d at ip %zu\n",
                current_instruction ? current_instruction->tag : -1, ip);
        return ERR;

    // Should be unreachable
    return ERR;
}

// Reference C implementation for z-score calculation
err_t compute_zscore_ref(const float* x_in, float* z_out, size_t N) {
    // Assume N > 0
    // Assume x_in and z_out are valid pointers

    // Calculate sum and sum of squares
    double sum = 0.0;
    double sum_sq = 0.0;
    for (size_t i = 0; i < N; ++i) {
        sum += x_in[i];
        sum_sq += x_in[i] * x_in[i];
    }

    // Calculate mean and mean_x_sq
    float mean = (float)(sum / N);
    float mean_x_sq = (float)(sum_sq / N);

    // Calculate variance and standard deviation
    float variance = mean_x_sq - mean * mean;
    // Clamp variance to non-negative before sqrt, handles potential floating point inaccuracies
    // Match interpreter behavior (no explicit zero check, relies on sqrt returning 0 or NaN)
    // if (variance < 0.0f) variance = 0.0f;
    float stddev = sqrtf(variance);

    // Handle potential division by zero if stddev is very small or zero
    // For direct comparison, we mimic the interpreter's lack of explicit check here.
    // A robust implementation would check `stddev` before dividing.
    // const float epsilon = 1e-8f;
    // if (stddev < epsilon) {
    //     // Handle zero stddev case (e.g., set all z_out to 0)
    //     for (size_t i = 0; i < N; ++i) {
    //         z_out[i] = 0.0f;
    //     }
    // } else {
        // Calculate z-scores
        for (size_t i = 0; i < N; ++i) {
            z_out[i] = (x_in[i] - mean) / stddev;
        }
    // }

    return OK;
}

// Benchmark function for z-score
static void benchmark_zscore(const float* x_in, float* z_out_interp, float* z_out_ref, size_t N, int iterations, const zscore_program_t* program) {
    struct timespec start_time, end_time;
    double total_time_interp = 0.0;
    double total_time_ref = 0.0;

    printf("\nRunning z-score benchmark (%d iterations)...\n", iterations);

    // Benchmark interpreted version
    for (int i = 0; i < iterations; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        compute_zscore_compiled(x_in, z_out_interp, program); // Use compiled program
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        total_time_interp += (end_time.tv_sec - start_time.tv_sec) +
                             (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    }

    // Benchmark reference C version
    for (int i = 0; i < iterations; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        compute_zscore_ref(x_in, z_out_ref, N);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        total_time_ref += (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
    }

    printf("Average time for interpreted z-score: %.6f ms\n", (total_time_interp / iterations) * 1000.0);
    printf("Average time for reference C z-score:  %.6f ms\n", (total_time_ref / iterations) * 1000.0);
    if (total_time_ref > 0) { // Avoid division by zero
        printf("Interpreted version is %.2fx slower than reference C version.\n", total_time_interp / total_time_ref);
    }
}

// Helper function to run a simple sequence (useful for single ops)
static err_t run_simple_instruction(instruction_t* instruction) {
    instruction_t seq[] = { *instruction, { .tag = OP_HALT } };
    return run_interpreter(seq); // Removed count
}

// This function now builds the instruction program.
zscore_program_t build_zscore_instructions(size_t N) {
    zscore_program_t program = {0}; // Initialize all pointers to NULL

    // Assume N > 0 and N % VECTOR_SIZE == 0

    size_t num_chunks = N / VECTOR_SIZE;
    const int chunk_byte_inc = VECTOR_SIZE * sizeof(float);

    // Scalar values will be stored in program.main_sequence[0].args.data_vec8.data
    // [0]: N_float, [1]: partial_sum, [2]: partial_sum_sq, [3]: mean,
    // [4]: mean_x_sq, [5]: variance, [6]: stddev, [7]: mean_squared

    // --- Heap allocate all instruction sequences and store in program struct ---

    // Define main_sequence structure on stack first
    instruction_t main_sequence_stack[] = {
        { .tag = OP_DATA_VEC8 }, // Index 0: Scalar storage
        { .tag = OP_DATA_VEC8 }, // Index 1: temp_vec_A (sum accumulator)
        { .tag = OP_DATA_VEC8 }, // Index 2: temp_vec_B (sum_sq accumulator / stddev_vec)
        { .tag = OP_DATA_VEC8 }, // Index 3: temp_vec_C (x_chunk^2 / (x_chunk - mean_vec))
        { .tag = OP_BLOCK, .args.block = { .instructions = NULL } }, // Index 4: Phase 1 block
        { .tag = OP_BLOCK, .args.block = { .instructions = NULL } }, // Index 5: Phase 2 block
        { .tag = OP_BLOCK, .args.block = { .instructions = NULL } }, // Index 6: Phase 3 block
        { .tag = OP_BLOCK, .args.block = { .instructions = NULL } }, // Index 7: Phase 4 block
        { .tag = OP_HALT }                                           // Index 8
    };
    program.main_sequence = (instruction_t*)malloc(sizeof(main_sequence_stack));
    memcpy(program.main_sequence, main_sequence_stack, sizeof(main_sequence_stack));

    // Initialize scalar storage. OP_DATA_VEC8 no longer auto-zeros.
    // Zero out all scalar storage initially.
    for(int i=0; i < VECTOR_SIZE; ++i) program.main_sequence[0].args.data_vec8.data[i] = 0.0f;
    program.main_sequence[0].args.data_vec8.data[0] = (float)N; // N_float

    // temp_vec_A, B, C (main_sequence[1,2,3]) will be zeroed by OP_ZERO_VEC8 in Phase 1
    // or have their contents overwritten by operations like OP_FILL_VEC8 in Phase 4.

    // Phase 1: Calculate Mean and Mean of Squares
    // Note: x_in pointers (src1_data, src2_data) will be patched by compute_zscore_compiled
    instruction_t phase1_loop_body_stack[] = {
        // temp_vec_C (idx 3) = x_chunk * x_chunk
        { .tag = OP_MUL_VEC8, .args.vec8_op = { .dst_data = program.main_sequence[3].args.data_vec8.data, .src1_data = NULL /*patched*/, .src2_data = NULL /*patched*/ } },
        // temp_vec_A (idx 1) = temp_vec_A + x_chunk
        { .tag = OP_ADD_VEC8, .args.vec8_op = { .dst_data = program.main_sequence[1].args.data_vec8.data, .src1_data = program.main_sequence[1].args.data_vec8.data, .src2_data = NULL /*patched*/ } },
        // temp_vec_B (idx 2) = temp_vec_B + temp_vec_C (idx 3)
        { .tag = OP_ADD_VEC8, .args.vec8_op = { .dst_data = program.main_sequence[2].args.data_vec8.data, .src1_data = program.main_sequence[2].args.data_vec8.data, .src2_data = program.main_sequence[3].args.data_vec8.data } },
        { .tag = OP_PINC, .args.pinc = { .p = NULL, .inc = chunk_byte_inc } },
        { .tag = OP_PINC, .args.pinc = { .p = NULL, .inc = chunk_byte_inc } },
        { .tag = OP_PINC, .args.pinc = { .p = NULL, .inc = chunk_byte_inc } },
        { .tag = OP_RETURN }
    };
    program.phase1_loop_body = (instruction_t*)malloc(sizeof(phase1_loop_body_stack));
    memcpy(program.phase1_loop_body, phase1_loop_body_stack, sizeof(phase1_loop_body_stack));
    program.phase1_loop_body[3].args.pinc.p = (void**)&program.phase1_loop_body[0].args.vec8_op.src1_data;
    program.phase1_loop_body[4].args.pinc.p = (void**)&program.phase1_loop_body[0].args.vec8_op.src2_data;
    program.phase1_loop_body[5].args.pinc.p = (void**)&program.phase1_loop_body[1].args.vec8_op.src2_data;

    instruction_t phase1_instructions_stack[] = {
        // Zero sum accumulator (temp_vec_A at program.main_sequence[1])
        { .tag = OP_ZERO_VEC8, .args.vec8_op = { .dst_data = program.main_sequence[1].args.data_vec8.data } },
        // Zero sum_sq accumulator (temp_vec_B at program.main_sequence[2])
        { .tag = OP_ZERO_VEC8, .args.vec8_op = { .dst_data = program.main_sequence[2].args.data_vec8.data } },
        { .tag = OP_LOOP_START, .args.loop_start = { .count = num_chunks } },
        { .tag = OP_BLOCK, .args.block = { .instructions = program.phase1_loop_body } },
        { .tag = OP_LOOP_END },
        // partial_sum (scalar_storage[1]) = sum(temp_vec_A (idx 1))
        { .tag = OP_SUM_VEC8, .args.sum_vec8 = { .src1_data = program.main_sequence[1].args.data_vec8.data, .result_ptr = &program.main_sequence[0].args.data_vec8.data[1] } },
        // partial_sum_sq (scalar_storage[2]) = sum(temp_vec_B (idx 2))
        { .tag = OP_SUM_VEC8, .args.sum_vec8 = { .src1_data = program.main_sequence[2].args.data_vec8.data, .result_ptr = &program.main_sequence[0].args.data_vec8.data[2] } },
        // partial_sum (scalar_storage[1]) /= N_float (scalar_storage[0])
        { .tag = OP_DIV_SCALAR, .args.scalar_binary = { .dst_ptr = &program.main_sequence[0].args.data_vec8.data[1], .src_ptr = &program.main_sequence[0].args.data_vec8.data[0] } },
        // partial_sum_sq (scalar_storage[2]) /= N_float (scalar_storage[0])
        { .tag = OP_DIV_SCALAR, .args.scalar_binary = { .dst_ptr = &program.main_sequence[0].args.data_vec8.data[2], .src_ptr = &program.main_sequence[0].args.data_vec8.data[0] } },
        // mean (scalar_storage[3]) = partial_sum (scalar_storage[1])
        { .tag = OP_COPY_SCALAR, .args.scalar_unary = { .dst_ptr = &program.main_sequence[0].args.data_vec8.data[3], .src_ptr = &program.main_sequence[0].args.data_vec8.data[1] } },
        // mean_x_sq (scalar_storage[4]) = partial_sum_sq (scalar_storage[2])
        { .tag = OP_COPY_SCALAR, .args.scalar_unary = { .dst_ptr = &program.main_sequence[0].args.data_vec8.data[4], .src_ptr = &program.main_sequence[0].args.data_vec8.data[2] } },
        { .tag = OP_RETURN }
    };
    program.phase1_instructions = (instruction_t*)malloc(sizeof(phase1_instructions_stack));
    memcpy(program.phase1_instructions, phase1_instructions_stack, sizeof(phase1_instructions_stack));
    program.main_sequence[4].args.block.instructions = program.phase1_instructions; // Index updated

    // Phase 2: Calculate Variance
    instruction_t phase2_instructions_stack[] = {
        // mean_squared (scalar_storage[7]) = mean (scalar_storage[3])
        { .tag = OP_COPY_SCALAR, .args.scalar_unary = { .dst_ptr = &program.main_sequence[0].args.data_vec8.data[7], .src_ptr = &program.main_sequence[0].args.data_vec8.data[3] } },
        // mean_squared (scalar_storage[7]) *= mean (scalar_storage[3])
        { .tag = OP_MUL_SCALAR, .args.scalar_binary = { .dst_ptr = &program.main_sequence[0].args.data_vec8.data[7], .src_ptr = &program.main_sequence[0].args.data_vec8.data[3] } },
        // variance (scalar_storage[5]) = mean_x_sq (scalar_storage[4])
        { .tag = OP_COPY_SCALAR, .args.scalar_unary = { .dst_ptr = &program.main_sequence[0].args.data_vec8.data[5], .src_ptr = &program.main_sequence[0].args.data_vec8.data[4] } },
        // variance (scalar_storage[5]) -= mean_squared (scalar_storage[7])
        { .tag = OP_SUB_SCALAR, .args.scalar_binary = { .dst_ptr = &program.main_sequence[0].args.data_vec8.data[5], .src_ptr = &program.main_sequence[0].args.data_vec8.data[7] } },
        { .tag = OP_RETURN }
    };
    program.phase2_instructions = (instruction_t*)malloc(sizeof(phase2_instructions_stack));
    memcpy(program.phase2_instructions, phase2_instructions_stack, sizeof(phase2_instructions_stack));
    program.main_sequence[5].args.block.instructions = program.phase2_instructions; // Index updated

    // Phase 3: Calculate Standard Deviation
    instruction_t phase3_instructions_stack[] = {
        // stddev (scalar_storage[6]) = sqrt(variance (scalar_storage[5]))
        { .tag = OP_SQRT_SCALAR, .args.scalar_unary = { .dst_ptr = &program.main_sequence[0].args.data_vec8.data[6], .src_ptr = &program.main_sequence[0].args.data_vec8.data[5] } },
        { .tag = OP_RETURN }
    };
    program.phase3_instructions = (instruction_t*)malloc(sizeof(phase3_instructions_stack));
    memcpy(program.phase3_instructions, phase3_instructions_stack, sizeof(phase3_instructions_stack));
    program.main_sequence[6].args.block.instructions = program.phase3_instructions; // Index updated

    // Phase 4: Calculate Z-Scores
    // Note: x_in (src1_data) and z_out (dst_data) pointers will be patched by compute_zscore_compiled
    instruction_t phase4_loop_body_stack[] = {
        // temp_vec_C (idx 3) = x_chunk - temp_vec_A (mean_vec, idx 1)
        { .tag = OP_SUB_VEC8, .args.vec8_op = { .dst_data = program.main_sequence[3].args.data_vec8.data, .src1_data = NULL /*patched*/, .src2_data = program.main_sequence[1].args.data_vec8.data } },
        // z_chunk = temp_vec_C (idx 3) / temp_vec_B (stddev_vec, idx 2)
        { .tag = OP_DIV_VEC8, .args.vec8_op = { .dst_data = NULL /*patched*/, .src1_data = program.main_sequence[3].args.data_vec8.data, .src2_data = program.main_sequence[2].args.data_vec8.data } },
        { .tag = OP_PINC, .args.pinc = { .p = NULL, .inc = chunk_byte_inc } },
        { .tag = OP_PINC, .args.pinc = { .p = NULL, .inc = chunk_byte_inc } },
        { .tag = OP_RETURN }
    };
    program.phase4_loop_body = (instruction_t*)malloc(sizeof(phase4_loop_body_stack));
    memcpy(program.phase4_loop_body, phase4_loop_body_stack, sizeof(phase4_loop_body_stack));
    program.phase4_loop_body[2].args.pinc.p = (void**)&program.phase4_loop_body[0].args.vec8_op.src1_data;
    program.phase4_loop_body[3].args.pinc.p = (void**)&program.phase4_loop_body[1].args.vec8_op.dst_data;

    instruction_t phase4_instructions_stack[] = {
        // temp_vec_A (idx 1) = mean (scalar_storage[3]) (scalar fill)
        { .tag = OP_FILL_VEC8, .args.fill_vec8 = { .dst_data = program.main_sequence[1].args.data_vec8.data, .scalar_ptr = &program.main_sequence[0].args.data_vec8.data[3] } },
        // temp_vec_B (idx 2) = stddev (scalar_storage[6]) (scalar fill)
        { .tag = OP_FILL_VEC8, .args.fill_vec8 = { .dst_data = program.main_sequence[2].args.data_vec8.data, .scalar_ptr = &program.main_sequence[0].args.data_vec8.data[6] } },
        // Loop
        { .tag = OP_LOOP_START, .args.loop_start = { .count = num_chunks } },
        { .tag = OP_BLOCK, .args.block = { .instructions = program.phase4_loop_body } },
        { .tag = OP_LOOP_END },
        { .tag = OP_RETURN } // End of Phase 4 block
    };
    program.phase4_instructions = (instruction_t*)malloc(sizeof(phase4_instructions_stack));
    memcpy(program.phase4_instructions, phase4_instructions_stack, sizeof(phase4_instructions_stack));
    program.main_sequence[7].args.block.instructions = program.phase4_instructions; // Index updated

    // Memory is not freed here; it's part of the returned program.
    // The caller is responsible for freeing the program using free_zscore_instructions.
    return program;
}

// Function to execute the pre-built z-score program
err_t compute_zscore_compiled(const float* x_in, float* z_out, const zscore_program_t* program) {
    if (!program || !program->main_sequence ||
        !program->phase1_loop_body || !program->phase4_loop_body) {
        // A more robust check would ensure all program pointers are non-NULL.
        fprintf(stderr, "Error: Invalid z-score program provided to compute_zscore_compiled.\n");
        return ERR;
    }

    // Patch input and output pointers for the current run.
    // These pointers are initially set to NULL by build_zscore_instructions
    // and are targeted by OP_PINC within the loop bodies.

    // Phase 1 loop body:
    // Instruction 0: temp_vec_C = x_chunk * x_chunk
    program->phase1_loop_body[0].args.vec8_op.src1_data = (float*)x_in; // Patched
    program->phase1_loop_body[0].args.vec8_op.src2_data = (float*)x_in; // Patched
    // Instruction 1: temp_vec_A = temp_vec_A + x_chunk
    program->phase1_loop_body[1].args.vec8_op.src2_data = (float*)x_in; // Patched

    // Phase 4 loop body:
    // Instruction 0: temp_vec_C = x_chunk - temp_vec_A (mean_vec)
    program->phase4_loop_body[0].args.vec8_op.src1_data = (float*)x_in; // Patched
    // Instruction 1: z_chunk = temp_vec_C / temp_vec_B (stddev_vec)
    program->phase4_loop_body[1].args.vec8_op.dst_data = z_out;         // Patched
    
    // N_float and num_chunks are already baked into the program by build_zscore_instructions.
    // Temporary vectors (temp_vec_A, B, C) and scalar storage are part of program->main_sequence
    // and are managed by the instruction stream (e.g. OP_ZERO_VEC8, OP_FILL_VEC8).

    return run_interpreter(program->main_sequence);
}

// Function to free the allocated z-score program instructions
void free_zscore_instructions(zscore_program_t* program) {
    if (!program) return;
    free(program->main_sequence);
    free(program->phase1_loop_body);
    free(program->phase1_instructions);
    free(program->phase2_instructions);
    free(program->phase3_instructions);
    free(program->phase4_loop_body);
    free(program->phase4_instructions);
    // Set pointers to NULL to prevent double free if called again, and to indicate freed state.
    program->main_sequence = NULL;
    program->phase1_loop_body = NULL;
    program->phase1_instructions = NULL;
    program->phase2_instructions = NULL;
    program->phase3_instructions = NULL;
    program->phase4_loop_body = NULL;
    program->phase4_instructions = NULL;
}


int main() {
    const size_t TEST_N = VECTOR_SIZE * 256; // Ensure multiple of VECTOR_SIZE, increased size
    const float epsilon = 1e-6f; // Tolerance for float comparison
    const size_t alignment = 64; // Desired alignment in bytes

    // Allocate memory using aligned_alloc
    // Ensure size is a multiple of alignment (should be fine if TEST_N * sizeof(float) is)
    // Note: aligned_alloc requires C11 or later. Size must be multiple of alignment.
    size_t total_bytes = TEST_N * sizeof(float);
    // Ensure total_bytes is a multiple of alignment for aligned_alloc requirement
    size_t aligned_total_bytes = (total_bytes + alignment - 1) / alignment * alignment;

    float* x_in = (float*)aligned_alloc(alignment, aligned_total_bytes);
    float* z_out_interp = (float*)aligned_alloc(alignment, aligned_total_bytes);
    float* z_out_ref = (float*)aligned_alloc(alignment, aligned_total_bytes);


    if (!x_in || !z_out_interp || !z_out_ref) {
        fprintf(stderr, "Aligned memory allocation failed!\n");
        // Use standard free for memory allocated with aligned_alloc
        free(x_in);
        free(z_out_interp);
        free(z_out_ref);
        return 1;
    }

    // Initialize input data (e.g., simple sequence)
    for (size_t i = 0; i < TEST_N; ++i) {
        x_in[i] = (float)(i + 1);
    }
    // Add a constant case to test zero stddev if desired
    // for (size_t i = 0; i < TEST_N; ++i) { x_in[i] = 5.0f; }

    printf("Building z-score program...\n");
    zscore_program_t z_program = build_zscore_instructions(TEST_N);
    // Basic check if build failed (e.g., malloc issues if we added error checks in build_zscore_instructions)
    if (!z_program.main_sequence) {
        fprintf(stderr, "Failed to build z-score program.\n");
        // Note: A more robust build_zscore_instructions would return an error or set a flag.
        // free_zscore_instructions(&z_program); // Attempt to free any partial allocations
        free(x_in);
        free(z_out_interp);
        free(z_out_ref);
        return 1;
    }

    printf("Running interpreter z-score calculation (compiled program)...\n");
    if (compute_zscore_compiled(x_in, z_out_interp, &z_program) != OK) {
        fprintf(stderr, "Interpreter z-score calculation failed!\n");
        free_zscore_instructions(&z_program);
        free(x_in);
        free(z_out_interp);
        free(z_out_ref);
        return 1;
    }

    printf("Running reference C z-score calculation...\n");
    if (compute_zscore_ref(x_in, z_out_ref, TEST_N) != OK) {
        fprintf(stderr, "Reference C z-score calculation failed!\n");
        free(x_in);
        free(z_out_interp);
        free(z_out_ref);
        return 1;
    }

    // Compare results
    printf("Comparing results...\n");
    int mismatch_found = 0;
    for (size_t i = 0; i < TEST_N; ++i) {
        if (fabsf(z_out_interp[i] - z_out_ref[i]) > epsilon) {
            fprintf(stderr, "Mismatch found at index %zu: interp=%.8f, ref=%.8f, diff=%.8f\n",
                    i, z_out_interp[i], z_out_ref[i], fabsf(z_out_interp[i] - z_out_ref[i]));
            mismatch_found = 1;
            // break; // Optional: stop at first mismatch
        }
        // Optional: Print some values for verification
        // if (i < 10 || i >= TEST_N - 10) {
        //     printf("  idx %zu: interp=%.6f, ref=%.6f\n", i, z_out_interp[i], z_out_ref[i]);
        // }
    }

    // Print final status
    if (mismatch_found) {
        printf("TEST FAILED: Results do not match within tolerance %.8f.\n", epsilon);
    } else {
        printf("TEST PASSED: Results match within tolerance %.8f.\n", epsilon);
    }

    // Run benchmark
    int benchmark_iterations = 1000; // Number of iterations for benchmark, increased iterations
    benchmark_zscore(x_in, z_out_interp, z_out_ref, TEST_N, benchmark_iterations, &z_program);

    // Cleanup
    free_zscore_instructions(&z_program);
    free(x_in);
    free(z_out_interp);
    free(z_out_ref);

    return mismatch_found; // Return 0 on success, 1 on failure
}
