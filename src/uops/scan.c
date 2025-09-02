#include "uops.h"

INLINE static err_t scan_inline(
    scan_t s
) {
    // Basic validation
    if (!s.f || !s.x || !s.y || s.n == 0 || s.sz == 0) {
        return ERR; // Invalid arguments
    }

    err_t e = OK;
    scan_f_t a = {.c=s.c, .x=s.x, .y=s.y};
    for (size_t i = 0; i < s.n; i++)
    {
      e |= s.f(a);
      a.x = ((char*) a.x) + s.sz;
      a.y = ((char*) a.y) + s.sz;
    }
    return e;
}

