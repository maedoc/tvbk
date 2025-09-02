#include "uops.h"

INLINE static err_t seq_inline(
    seq_t s
) {
    // Basic validation
    if (!s.ops || !s.args || s.n == 0) {
        return ERR; // Invalid arguments
    }

    err_t e = OK;
    for (size_t i = 0; i < s.n; i++) {
        // Check if the current op function pointer is valid before calling
        if (!s.ops[i]) {
            return ERR; // Invalid operation function pointer in the sequence
        }
        // Note: We don't explicitly check s.args[i] for NULL here,
        // as individual ops are responsible for validating their own arguments.
        e |= s.ops[i](s.args[i]); // Call the op with its corresponding argument
        if (e != OK) return e;    // Early exit on error
    }
    return e;
}
