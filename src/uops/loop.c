#include "uops.h"

INLINE static err_t loop_inline(
    loop_t l
) {
    // Basic validation
    if (!l.op || !l.inc) { // n can be 0 for an empty loop
        return ERR; // Invalid arguments
    }

    err_t e = OK;
    for (size_t i = 0; i < l.n; i++)
    {
      e |= l.op(l.arg); // Pass the provided argument to the op
      if (e != OK) return e; // Early exit on error
      (*l.inc)++; // Increment the integer pointed to by inc
    }
    return e;
}
