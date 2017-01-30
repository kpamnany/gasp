/*  gasp -- global address space toolbox

    distributed dynamic scheduler debugging and profiling
 */


#ifndef _DTREE_DEBUG_H
#define _DTREE_DEBUG_H

#include <stdint.h>

#ifdef DEBUG_DTREE
#ifndef TRACE_DTREE
#define TRACE_DTREE        1
#endif
#endif

#ifdef TRACE_DTREE

#if TRACE_DTREE == 1
#define DTREE_TRACE(dt,x...)          \
    if ((dt)->g->rank == 0 || (dt)->g->rank == (dt)->g->nranks-1) \
        fprintf(stderr, x)
#elif TRACE_DTREE == 2
#define DTREE_TRACE(dt,x...)          \
    if ((dt)->g->rank < 18 || (dt)->g->rank > (dt)->g->nranks-19) \
        fprintf(stderr, x)
#elif TRACE_DTREE == 3
#define DTREE_TRACE(dt,x...)          \
    fprintf(stderr, x)
#else
#define DTREE_TRACE(dt,x...)
#endif

#else

#define DTREE_TRACE(x...)

#endif

#ifdef PROFILE_DTREE
enum {
    DTREE_TIME_GETWORK, DTREE_TIME_MPIWAIT, DTREE_TIME_MPISEND, DTREE_TIME_RUN,
    DTREE_NTIMES
};

char *dtree_times_names[] = {
    "getwork", "mpiwait", "mpisend", "runtree", ""
};

typedef struct dtree_thread_timing_tag {
    uint64_t last, min, max, total, count;
} dtree_thread_timing_t;
#endif


#endif /* _DTREE_DEBUG_H */

