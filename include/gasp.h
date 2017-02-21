/*  gasp -- global address space toolbox

    interface definition for global arrays and dtree
 */

#ifndef _GASP_H
#define _GASP_H

#include <mpi.h>
#include <stdint.h>
#include "util.h"
#include "garray_debug.h"
#include "dtree_debug.h"
#include "log.h"


/* gasp handle */
typedef struct gasp_tag {
    int nranks, rank;
    log_t glog, dlog;
} gasp_t;


/* global array */
typedef struct garray_tag {
    gasp_t     *g;
    int64_t     num_elems, *chunks, elem_size, nextra_elems, nelems_per_rank,
                nlocal_elems;
    int8_t      *buffer;
    MPI_Win     win;

#ifdef PROFILE_GARRAY
    garray_thread_timing_t **times;
#endif
} garray_t;


/* dtree */
typedef struct dtree_tag {
    gasp_t             *g;

    /* tree structure */
    int8_t              num_levels, my_level;
    int                 parent, *children, num_children;
    double              tot_children;

    /* MPI info */
    int16_t             *children_req_bufs;
    MPI_Request         parent_req, *children_reqs;

    /* work distribution policy */
    int16_t             parents_work;
    double              first, rest;
    double              *distrib_fractions;
    int16_t             min_distrib;

    /* work items */
    int64_t             first_work_item, last_work_item, next_work_item;
    int64_t volatile    work_lock __attribute((aligned(8)));

    /* for heterogeneous clusters */
    double              rank_mul;

    /* concurrent calls in from how many threads? */
    int                 num_threads;
    int                 (*threadid)();
#ifdef PROFILE_DTREE
    dtree_thread_timing_t **times;
#endif

} dtree_t;


/* gasp interface
 */

int      gasp_init(int ac, char **av, gasp_t **g);
void     gasp_shutdown(gasp_t *g);
void     gasp_sync();

/* number of participating ranks */
int      gasp_nranks();

/* this rank's unique identifier */
int      gasp_rank();

/* global array */
int      garray_create(gasp_t *g, int64_t num_elems, int64_t elem_size,
                       int64_t *chunks, garray_t **ga);
void     garray_destroy(garray_t *ga);

int64_t  garray_length(garray_t *ga);
int64_t  garray_elemsize(garray_t *ga);

int      garray_get(garray_t *ga, int64_t lo, int64_t hi, void *buf);
int      garray_put(garray_t *ga, int64_t lo, int64_t hi, void *buf);

int      garray_distribution(garray_t *ga, int rank, int64_t *lo, int64_t *hi);
int      garray_access(garray_t *ga, int64_t lo, int64_t hi, void **buf);

void     garray_flush(garray_t *ga);

/* dtree */
int      dtree_create(gasp_t *g, int fan_out, int64_t num_work_items,
                      int can_parent, int parents_work, double rank_mul,
                      int num_threads, int (*threadid)(),
                      double first, double rest, int16_t min_distrib,
                      dtree_t **dt, int *is_parent);
void     dtree_destroy(dtree_t *dt);

/* call to get initial work allocation; before dtree_run() is called */
int64_t  dtree_initwork(dtree_t *dt, int64_t *first_item, int64_t *last_item);

/* get a block of work */
int64_t  dtree_getwork(dtree_t *dt, int64_t *first_item, int64_t *last_item);

/* call from a thread repeatedly until it returns 0 */
int      dtree_run(dtree_t *dt);

/* utility helpers */
uint64_t rdtsc();
void     cpu_pause();
void     start_sde_tracing();
void     stop_sde_tracing();

#endif  /* _GASP_H */

