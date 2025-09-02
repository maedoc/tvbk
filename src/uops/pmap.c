#include "uops.h"
#include <stdlib.h>     // For malloc, free
#include <stdio.h>      // For fprintf, stderr
#include <stddef.h>     // For size_t
#include "tinycthread.h" // Use tinycthread for threading

typedef struct { op_t op; void *arg; } work_t;

static int work_f(work_t *w) {
  return w->op(w->arg) == OK ? thrd_success : thrd_error;
}

// Simplified pmap implementation: one thread per task
INLINE static err_t pmap_inline(pmap_t p) {
    if (!p.ops || !p.args || p.n == 0) {
        fprintf(stderr, "[pmap_inline] Error: Invalid arguments (ops, args, or n=0).\n");
        return ERR;
    }
    thrd_t *threads = malloc(p.n * sizeof(thrd_t));
    work_t *works = malloc(p.n * sizeof(work_t));
    // start
    for (size_t i = 0; i < p.n; ++i) {
        works[i] = (work_t) {.op = p.ops[i], .arg=p.args[i]};
        thrd_create(threads+i, (thrd_start_t) work_f, works+i);
    }
    // wait
    err_t ok = OK;
    for (int i = 0; i < p.n; ++i) {
        int thrd_ok = thrd_error;
        thrd_join(threads[i], &thrd_ok);
        ok |= thrd_ok == thrd_error;
    }
    // cleanup
    free(threads);
    free(works);
    return ok;
}
