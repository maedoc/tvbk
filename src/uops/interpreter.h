#pragma once

#include "uops.h" // For UOPS_API, err_t, size_t
#include <stdint.h>
#include <stddef.h>

#define MAX_INTERP_STACK 32 // Maximum nesting depth for loops

// --- Operation Types ---
typedef enum {
    OP_NOOP,        // Does nothing, advances IP
    OP_HALT,        // Stops the interpreter successfully
    OP_SINF_SCALAR, // Applies sinf to a single float*
    OP_SINF_VECTOR, // Applies sinf element-wise to a float vector
    OP_LOOP_START,  // Marks the beginning of a loop N times
    OP_LOOP_END,    // Marks the end of a loop, handles counter and jump
    OP_SEQ,         // Executes a sequence of C function pointers
    OP_BLOCK,       // Executes a nested block of native instructions
    OP_RETURN       // Returns from a nested block execution
    // Add other specific operations here
} op_type_t;

// --- Argument Structs for Specific Operations ---

typedef struct {
    float* val_ptr; // Pointer to the float value
} op_args_sinf_scalar_t;

typedef struct {
    float* data;    // Pointer to vector data
    size_t len;     // Length of the vector
} op_args_sinf_vector_t;

typedef struct {
    size_t count;   // Number of iterations
} op_args_loop_start_t;

typedef struct {
    const op_t* ops; // Array of C function pointers to execute
    void** args;     // Array of arguments for those functions
    size_t n;        // Number of functions in the sequence
} op_args_seq_t;

// Forward declaration for instruction_t needed by op_args_block_t
typedef struct instruction_s instruction_t;

typedef struct {
    struct instruction_s* instructions;       // Pointer to the block's instruction array (use struct tag)
    size_t                num_instructions; // Number of instructions in the block
} op_args_block_t;


// --- Union of All Possible Arguments ---
typedef union {
    op_args_sinf_scalar_t sinf_scalar;
    op_args_sinf_vector_t sinf_vector;
    op_args_loop_start_t  loop_start;
    op_args_seq_t         seq;
    op_args_block_t       block;
    // Add other argument structs here
    // No args needed for NOOP, HALT, LOOP_END, RETURN
} op_args_union_t;

// --- The Instruction Structure (Tagged Union) ---
// Now define the actual struct using the tag declared earlier
struct instruction_s {
    op_type_t       tag;
    op_args_union_t args;
};
// The typedef instruction_t already exists from the forward declaration.

// --- Interpreter Loop Stack (for OP_LOOP) ---
typedef struct {
    size_t loop_counter; // Remaining iterations
    size_t loop_start_ip; // IP of the instruction *after* OP_LOOP_START
} loop_stack_entry_t;

// --- Interpreter State Stack (for OP_BLOCK/OP_RETURN) ---
typedef struct {
    instruction_t* parent_instructions;       // Caller's instruction array
    size_t         parent_num_instructions; // Caller's instruction count
    size_t         return_ip;               // IP to return to in caller
} interp_state_stack_entry_t;

// --- Interpreter Function ---

/**
 * @brief Executes a sequence of instructions.
 *
 * @param instructions Pointer to the array of instructions.
 * @param num_instructions The number of instructions in the array.
 * @return OK on successful execution (reaching OP_HALT), ERR otherwise.
 */
UOPS_API err_t run_interpreter(instruction_t* instructions, size_t num_instructions);
