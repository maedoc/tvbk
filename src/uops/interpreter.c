#include "interpreter.h"
#include <math.h> // For sinf
#include <stdio.h> // For fprintf, stderr
#include <stddef.h> // For size_t

// Note: Assumes sinf is available via math.h

// Helper function to find the matching LOOP_END for a LOOP_START
// start_ip: The instruction pointer *after* the LOOP_START instruction.
// Returns the instruction pointer *after* the matching LOOP_END, or (size_t)-1 on error.
static size_t find_matching_loop_end(const instruction_t* instructions, size_t num_instructions, size_t start_ip) {
    size_t nesting_level = 1;
    size_t current_ip = start_ip;

    while (current_ip < num_instructions) {
        op_type_t tag = instructions[current_ip].tag;
        if (tag == OP_LOOP_START) {
            nesting_level++;
        } else if (tag == OP_LOOP_END) {
            nesting_level--;
            if (nesting_level == 0) {
                return current_ip + 1; // Found the matching end, return IP for instruction after it
            }
        }
        current_ip++;
    }

    // If we reach here, no matching LOOP_END was found
    fprintf(stderr, "[interp] error: unmatched loop start at ip %zu\n", start_ip - 1);
    return (size_t)-1; // Error indicator
}

UOPS_API err_t run_interpreter(instruction_t* instructions, size_t num_instructions) {
    if (!instructions || num_instructions == 0) {
        fprintf(stderr, "[interp] error: no instructions\n");
        return ERR;
    }

    size_t ip = 0; // Instruction Pointer, relative to current `instructions` array
    loop_stack_entry_t loop_stack[MAX_INTERP_STACK]; // For OP_LOOP/END
    int loop_sp = -1; // Loop Stack Pointer (-1 means empty)

    interp_state_stack_entry_t interp_state_stack[MAX_INTERP_STACK]; // For OP_BLOCK/RETURN
    int interp_sp = -1; // Interpreter State Stack Pointer (-1 means empty)

    // Need mutable pointers/counts for block execution context switching
    instruction_t* current_instructions = instructions;
    size_t current_num_instructions = num_instructions;

    while (ip < current_num_instructions) {
        const instruction_t* current_instruction = &current_instructions[ip];
        err_t op_result = OK;

        // printf("[interp] ip=%zu op=%d loop_sp=%d interp_sp=%d\n", ip, current_instruction->tag, loop_sp, interp_sp); // Debug

        switch (current_instruction->tag) {
            case OP_HALT:
                // printf("[interp] halt\n"); // Debug
                return OK;

            case OP_NOOP:
                ip++;
                break;

            case OP_SINF_SCALAR: {
                float* val_ptr = current_instruction->args.sinf_scalar.val_ptr;
                if (!val_ptr) {
                    fprintf(stderr, "[interp] error: sinf_scalar null ptr at ip %zu\n", ip);
                    return ERR;
                }
                *val_ptr = sinf(*val_ptr);
                ip++;
                break;
            }

            case OP_SINF_VECTOR: {
                float* data = current_instruction->args.sinf_vector.data;
                size_t len = current_instruction->args.sinf_vector.len;
                if (!data || len == 0) {
                     fprintf(stderr, "[interp] error: sinf_vector null/zero len at ip %zu\n", ip);
                    return ERR;
                }
                for (size_t i = 0; i < len; ++i) {
                    data[i] = sinf(data[i]);
                }
                ip++;
                break;
            }

            case OP_LOOP_START: {
                size_t count = current_instruction->args.loop_start.count;

                if (count == 0) {
                    // Skip the loop entirely
                    size_t end_ip = find_matching_loop_end(instructions, num_instructions, ip + 1);
                    if (end_ip == (size_t)-1) {
                        // Error message already printed by helper function
                        return ERR;
                    }
                    ip = end_ip; // Jump past the OP_LOOP_END
                    // printf("[interp] loop_start: skip zero-count loop, jump to ip=%zu\n", ip); // Debug
                } else {
                    // Push loop state onto the loop stack
                    if (loop_sp >= MAX_INTERP_STACK - 1) {
                        fprintf(stderr, "[interp] error: loop stack overflow at ip %zu\n", ip);
                        return ERR;
                    }
                    loop_sp++;
                    loop_stack[loop_sp].loop_counter = count;
                    loop_stack[loop_sp].loop_start_ip = ip + 1; // Jump back *after* LOOP_START
                    // printf("[interp] loop_start: push count=%zu start_ip=%zu loop_sp=%d\n", count, ip + 1, loop_sp); // Debug
                    ip++; // Move to the first instruction inside the loop
                }
                break;
            }

            case OP_LOOP_END: {
                if (loop_sp < 0) {
                    fprintf(stderr, "[interp] error: loop_end without loop_start at ip %zu\n", ip);
                    return ERR;
                }

                // Decrement counter *before* checking
                if (loop_stack[loop_sp].loop_counter > 0) {
                     loop_stack[loop_sp].loop_counter--;
                }

                // printf("[interp] loop_end: check count=%zu start_ip=%zu loop_sp=%d\n", loop_stack[loop_sp].loop_counter, loop_stack[loop_sp].loop_start_ip, loop_sp); // Debug

                if (loop_stack[loop_sp].loop_counter > 0) {
                    // Jump back
                    ip = loop_stack[loop_sp].loop_start_ip;
                    // printf("[interp] loop_end: jump back to %zu\n", ip); // Debug
                } else {
                    // Loop finished, pop stack and continue past LOOP_END
                    loop_sp--;
                    ip++;
                    // printf("[interp] loop_end: finished, pop loop_sp=%d ip=%zu\n", loop_sp, ip); // Debug
                }
                break;
            }

            case OP_SEQ: {
                const op_args_seq_t* seq_args = &current_instruction->args.seq;
                // Minimal error checking: Assume ops, args are valid and n > 0
                // if (!seq_args->ops || !seq_args->args || seq_args->n == 0) return ERR;

                err_t seq_result = OK;
                for (size_t i = 0; i < seq_args->n; ++i) {
                    // Minimal error checking: Assume ops[i] is valid
                    // if (!seq_args->ops[i]) return ERR;

                    seq_result = seq_args->ops[i](seq_args->args[i]);
                    if (seq_result != OK) {
                        // Propagate error immediately
                        fprintf(stderr, "[interp] error: OP_SEQ sub-operation at index %zu failed (ip %zu)\n", i, ip);
                        return ERR;
                    }
                }
                // If all sub-operations succeeded
                ip++;
                break;
            }

            case OP_BLOCK: {
                const op_args_block_t* block_args = &current_instruction->args.block;
                if (!block_args->instructions || block_args->num_instructions == 0) {
                    fprintf(stderr, "[interp] error: OP_BLOCK invalid instructions/num_instructions at ip %zu\n", ip);
                    return ERR;
                }
                if (interp_sp >= MAX_INTERP_STACK - 1) {
                    fprintf(stderr, "[interp] error: interpreter state stack overflow at ip %zu\n", ip);
                    return ERR;
                }

                // Push current state
                interp_sp++;
                interp_state_stack[interp_sp].parent_instructions = current_instructions;
                interp_state_stack[interp_sp].parent_num_instructions = current_num_instructions;
                interp_state_stack[interp_sp].return_ip = ip + 1; // Return to instruction *after* OP_BLOCK

                // Switch context to the block
                current_instructions = block_args->instructions;
                current_num_instructions = block_args->num_instructions;
                ip = 0; // Start executing the block from the beginning

                // printf("[interp] block_start: push return_ip=%zu interp_sp=%d\n", interp_state_stack[interp_sp].return_ip, interp_sp); // Debug
                break;
            }

            case OP_RETURN: {
                if (interp_sp < 0) {
                    fprintf(stderr, "[interp] error: OP_RETURN without OP_BLOCK (stack underflow) at ip %zu\n", ip);
                    return ERR;
                }

                // Pop state to return to caller
                current_instructions = interp_state_stack[interp_sp].parent_instructions;
                current_num_instructions = interp_state_stack[interp_sp].parent_num_instructions;
                ip = interp_state_stack[interp_sp].return_ip;
                interp_sp--;

                // printf("[interp] return: pop return_ip=%zu interp_sp=%d\n", ip, interp_sp); // Debug
                break;
            }

            default:
                fprintf(stderr, "[interp] error: unknown opcode %d at ip %zu\n", current_instruction->tag, ip);
                return ERR;
        }

        if (op_result != OK) { // Should not happen with current ops
             fprintf(stderr, "[interp] error: op failed at ip %zu\n", ip > 0 ? ip -1 : 0);
             return ERR;
        }

    } // end while loop

    // Check why the loop terminated
    if (ip == current_num_instructions) {
        if (interp_sp >= 0) {
            // Fell off the end of a block without OP_RETURN
            fprintf(stderr, "[interp] error: fell off end of block without OP_RETURN (ip %zu, block size %zu)\n", ip, current_num_instructions);
        } else {
            // Fell off the end of the top-level program without OP_HALT
            fprintf(stderr, "[interp] error: fell off end of top-level instructions without OP_HALT\n");
        }
    }
    // If we exited due to an error inside the loop, the error is already printed.
    // If we exited due to OP_HALT, OK was already returned.

    return ERR; // Default return if loop terminated unexpectedly or due to error
}
