/*  gasp -- global address space toolbox

    global arrays debugging and profiling
 */

#ifndef _GARRAY_DEBUG_H
#define _GARRAY_DEBUG_H

#include <stdint.h>

#ifdef DEBUG_GARRAY
#ifndef TRACE_GARRAY
#define TRACE_GARRAY    1
#endif
#endif

#ifdef TRACE_GARRAY

#if TRACE_GARRAY == 1
#define GARRAY_TRACE(g,x...)           \
    if ((g)->my_rank == 0 || (g)->my_rank == (g)->num_ranks-1)  \
        fprintf(stderr, x)
#elif TRACE_GARRAY == 2
#define GARRAY_TRACE(g,x...)           \
    if ((g)->my_rank < 18 || (g)->my_rank > (g)->num_ranks-19)  \
        fprintf(stderr, x)
#elif TRACE_GARRAY == 3
#define GARRAY_TRACE(g,x...)           \
    fprintf(stderr, x)
#else
#define GARRAY_TRACE(g,x...)
#endif

#else

#define GARRAY_TRACE(x...)

#endif

#ifdef PROFILE_GARRAY
enum {
    GARRAY_TIME_GET, GARRAY_TIME_PUT, GARRAY_TIME_SYNC,
    GARRAY_NTIMES
};

char *garray_times_names[] = {
    "get", "put", "sync", ""
};

typedef struct garray_thread_timing_tag {
    uint64_t last, min, max, total, count;
} garray_thread_timing_t;
#endif


#endif /* _GARRAY_DEBUG_H */
