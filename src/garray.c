/*  gasp -- global address space toolbox

    global arrays implementation
*/


#include <mpi.h>
#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <unistd.h>
#include <immintrin.h>

#include "gasp.h"


/*  garray_create()
 */
int garray_create(gasp_t *g, int64_t num_elems, int64_t elem_size,
            int64_t *chunks, garray_t **ga_)
{
    /* only regular distribution for now */
    assert(chunks == NULL);

    int r;
    garray_t *ga;
    r = posix_memalign((void **)&ga, 64, sizeof(garray_t));
    if (r != 0)
        return r;

    ga->g = g;

    /* distribution of elements */
    ldiv_t res = ldiv(num_elems, g->nranks);
    ga->nextra_elems = res.rem;
    ga->nelems_per_rank = ga->nlocal_elems = res.quot;
    if (g->rank < ga->nextra_elems)
        ++ga->nlocal_elems;

    /* fill in array info */
    ga->num_elems = num_elems;
    ga->elem_size = elem_size;

    /* allocate the array */
    MPI_Win_allocate(ga->nlocal_elems*elem_size, 1, MPI_INFO_NULL,
            MPI_COMM_WORLD, &ga->buffer, &ga->win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, ga->win);
    memset(ga->buffer, 0, ga->nlocal_elems * elem_size);
    MPI_Barrier(MPI_COMM_WORLD);

    *ga_ = ga;

    LOG_INFO(g->glog, "[%d] garray created, %" PRId64 " elements "
             "of size %" PRId64 "\n", g->rank, num_elems, elem_size);

    return 0;
}


/*  garray_destroy()
 */
void garray_destroy(garray_t *ga)
{
    garray_flush(ga);
    MPI_Win_unlock_all(ga->win);
    MPI_Win_free(&ga->win);

    LOG_INFO(ga->g->glog, "[%d] garray destroyed, %" PRId64 " elements "
             "of size %" PRId64 "\n", ga->g->rank, ga->num_elems,
             ga->elem_size);

    free(ga);
}


/*  garray_length()
 */
int64_t garray_length(garray_t *ga)
{
    return ga->num_elems;
}


/*  garary_elemsize()
 */
int64_t garray_elemsize(garray_t *ga)
{
    return ga->elem_size;
}


/*  calc_target()
 */
static void calc_target(garray_t *ga, int64_t gidx, int64_t *trank_, int64_t *tidx_)
{
    /* if this rank has no local elements, there are less than nranks
       elements, so the array index is the rank index */
    if (ga->nlocal_elems == 0) {
        *trank_ = gidx;
        *tidx_ = 0;
        return;
    }

    /* compute the target rank+idx */
    ldiv_t res = ldiv(gidx, ga->nlocal_elems);

    /* if the distribution is not perfectly even, we have to adjust
       the target rank+idx appropriately */
    if (ga->nextra_elems > 0) {
        int64_t trank = res.quot, tidx = res.rem;

        /* if i have an extra element... */
        if (ga->g->rank < ga->nextra_elems) {
            /* but the target does not */
            if (trank >= ga->nextra_elems) {
                /* then the target index has to be adjusted upwards */
                tidx += (trank - ga->nextra_elems);
                /* which may mean that the target rank does too */
                while (tidx >= (ga->nlocal_elems - 1)) {
                    ++trank;
                    tidx -= (ga->nlocal_elems - 1);
                }
            }
        }

        /* i don't have an extra element... */
        else {
            /* so adjust the target index downwards */
            tidx -= (trank < ga->nextra_elems ? trank : ga->nextra_elems);
            /* which may mean the target rank has to be adjusted too */
            while (tidx < 0) {
                --trank;
                tidx += ga->nelems_per_rank + (trank < ga->nextra_elems ? 1 : 0);
            }
        }

        res.quot = trank; res.rem = tidx;
    }

    LOG_DEBUG(ga->g->glog, "[%d] garray calc %" PRId64 ", target %ld.%ld\n",
              ga->g->rank, gidx, res.quot, res.rem);

    *trank_ = res.quot;
    *tidx_ = res.rem;
}


/*  garray_get()
 */
int garray_get(garray_t *ga, int64_t lo, int64_t hi, void *buf_)
{
    int64_t count = (hi - lo) + 1, length = count * ga->elem_size,
            tlorank, tloidx, thirank, thiidx, trank, tidx, n, oidx = 0;
    int8_t *buf = (int8_t *)buf_;

    if (lo < 0 || lo >= ga->num_elems || hi < 0 || hi >= ga->num_elems) {
        LOG_ERR(ga->g->glog, "[%d] garray get: invalid lo/hi index: "
                "%" PRId64 "-%" PRId64 " of %" PRId64 " elements\n",
                ga->g->rank, lo, hi, ga->num_elems);
        return -1;
    }

    calc_target(ga, lo, &tlorank, &tloidx);
    calc_target(ga, hi, &thirank, &thiidx);

    /* is all requested data on the same target? */
    if (tlorank == thirank) {
        LOG_DEBUG(ga->g->glog, "[%d] garray getting %" PRId64 "-%" PRId64 ", "
                  "single target %" PRId64 ".%" PRId64 "\n",
                  ga->g->rank, lo, hi, tlorank, tloidx);

        //MPI_Win_lock(MPI_LOCK_SHARED, tlorank, 0, ga->win);
        MPI_Get(buf, length, MPI_INT8_T,
                tlorank, (tloidx * ga->elem_size), length, MPI_INT8_T,
                ga->win);
        //MPI_Win_unlock(tlorank, ga->win);
        MPI_Win_flush_local(tlorank, ga->win);

        return 0;
    }

    /* get the data in the lo rank */
    n = ga->nelems_per_rank + (tlorank < ga->nextra_elems ? 1 : 0) - tloidx;

    LOG_DEBUG(ga->g->glog, "[%d] garray getting %" PRId64 " elements from "
              "%" PRId64 ".%" PRId64 "\n", ga->g->rank, n, tlorank, tloidx);

    //MPI_Win_lock(MPI_LOCK_SHARED, tlorank, 0, ga->win);
    MPI_Get(buf, (n * ga->elem_size), MPI_INT8_T,
            tlorank, (tloidx * ga->elem_size), (n * ga->elem_size), MPI_INT8_T,
            ga->win);
    //MPI_Win_unlock(tlorank, ga->win);

    oidx = (n * ga->elem_size);

    /* get the data in the in-between ranks */
    tidx = 0;
    for (trank = tlorank + 1;  trank < thirank;  ++trank) {
        n = ga->nelems_per_rank + (trank < ga->nextra_elems ? 1 : 0);

        LOG_DEBUG(ga->g->glog, "[%d] garray getting %" PRId64 " elements from "
                  "%" PRId64 ".%" PRId64 "\n", ga->g->rank, n, trank, tidx);

        //MPI_Win_lock(MPI_LOCK_SHARED, trank, 0, ga->win);
        MPI_Get(&buf[oidx], (n * ga->elem_size), MPI_INT8_T,
                trank, 0, (n * ga->elem_size), MPI_INT8_T,
                ga->win);
        //MPI_Win_unlock(trank, ga->win);

        oidx += (n * ga->elem_size);
    }

    /* get the data in the hi rank */
    n = thiidx + 1;

    LOG_DEBUG(ga->g->glog, "[%d] garray getting %" PRId64 " elements up to "
              "%" PRId64 ".%" PRId64 "\n", ga->g->rank, n, thirank, thiidx);

    //MPI_Win_lock(MPI_LOCK_SHARED, thirank, 0, ga->win);
    MPI_Get(&buf[oidx], (n * ga->elem_size), MPI_INT8_T,
            thirank, 0, (n * ga->elem_size), MPI_INT8_T,
            ga->win);
    //MPI_Win_unlock(thirank, ga->win);
    MPI_Win_flush_local_all(ga->win);

    return 0;
}


/*  garray_put()
 */
int garray_put(garray_t *ga, int64_t lo, int64_t hi, void *buf_)
{
    int64_t count = (hi - lo) + 1, length = count * ga->elem_size,
            tlorank, tloidx, thirank, thiidx, trank, tidx, n, oidx = 0;
    int8_t *buf = (int8_t *)buf_;

    if (lo < 0 || lo >= ga->num_elems || hi < 0 || hi >= ga->num_elems) {
        LOG_ERR(ga->g->glog, "[%d] garray put: invalid lo/hi index: "
                "%" PRId64 "-%" PRId64 " of %" PRId64 " elements\n",
                ga->g->rank, lo, hi, ga->num_elems);
        return -1;
    }

    calc_target(ga, lo, &tlorank, &tloidx);
    calc_target(ga, hi, &thirank, &thiidx);

    /* is all data going to the same target? */
    if (tlorank == thirank) {
        LOG_DEBUG(ga->g->glog, "[%d] garray put %" PRId64 "-%" PRId64 ", "
                  "single target %" PRId64 ".%" PRId64 "\n",
                  ga->g->rank, lo, hi, tlorank, tloidx);

        //MPI_Win_lock(MPI_LOCK_EXCLUSIVE, tlorank, 0, ga->win);
        MPI_Put(buf, length, MPI_INT8_T,
                tlorank, (tloidx * ga->elem_size), length, MPI_INT8_T,
                ga->win);
        //MPI_Win_unlock(tlorank, ga->win);
        MPI_Win_flush(tlorank, ga->win);

        return 0;
    }

    /* put the data into the lo rank */
    n = ga->nelems_per_rank + (tlorank < ga->nextra_elems ? 1 : 0) - tloidx;

    LOG_DEBUG(ga->g->glog, "[%d] garray putting %" PRId64 " elements into "
              "%" PRId64 ".%" PRId64 "\n", ga->g->rank, n, tlorank, tloidx);

    //MPI_Win_lock(MPI_LOCK_SHARED, tlorank, 0, ga->win);
    MPI_Put(buf, length, MPI_INT8_T,
            tlorank, (tloidx * ga->elem_size), (n * ga->elem_size), MPI_INT8_T,
            ga->win);
    //MPI_Win_unlock(tlorank, ga->win);

    oidx = (n*ga->elem_size);

    /* put the data into the in-between ranks */
    tidx = 0;
    for (trank = tlorank + 1;  trank < thirank;  ++trank) {
        n = ga->nelems_per_rank + (trank < ga->nextra_elems ? 1 : 0);

        LOG_DEBUG(ga->g->glog, "[%d] garray putting %" PRId64 " elements into "
                  "%" PRId64 ".%" PRId64 "\n", ga->g->rank, n, trank, tidx);

        //MPI_Win_lock(MPI_LOCK_EXCLUSIVE, trank, 0, ga->win);
        MPI_Put(&buf[oidx], (n * ga->elem_size), MPI_INT8_T,
                trank, 0, (n * ga->elem_size), MPI_INT8_T,
                ga->win);
        //MPI_Win_unlock(trank, ga->win);

        oidx += (n*ga->elem_size);
    }

    /* put the data into the hi rank */
    n = thiidx + 1;

    LOG_DEBUG(ga->g->glog, "[%d] garray putting %" PRId64 " elements up to "
              "%" PRId64 ".%" PRId64 "\n", ga->g->rank, n, thirank, thiidx);

    //MPI_Win_lock(MPI_LOCK_EXCLUSIVE, thirank, 0, ga->win);
    MPI_Put(&buf[oidx], (n * ga->elem_size), MPI_INT8_T,
            thirank, 0, (n * ga->elem_size), MPI_INT8_T,
            ga->win);
    //MPI_Win_unlock(thirank, ga->win);
    MPI_Win_flush_all(ga->win);

    return 0;
}


/*  garray_distribution()
 */
int garray_distribution(garray_t *ga, int rank, int64_t *lo, int64_t *hi)
{
    if (rank >= ga->g->nranks) {
        LOG_WARN(ga->g->glog, "[%d] garray distribution requested for rank %d out"
                  " of %d ranks\n", ga->g->rank, rank, ga->g->nranks);
        return -1;
    }

    int64_t a1, a2;
    if (rank < ga->nextra_elems) {
        a1 = rank;
        a2 = 1;
    }
    else {
        a1 = ga->nextra_elems;
        a2 = 0;
    }
    *lo = rank * ga->nelems_per_rank + a1;
    *hi = *lo + ga->nelems_per_rank + a2 - 1; /* inclusive */

    return 0;
}


/*  garray_access()
 */
int garray_access(garray_t *ga, int64_t lo, int64_t hi, void **buf)
{
    *buf = NULL;

    int64_t mylo, myhi;
    garray_distribution(ga, ga->g->rank, &mylo, &myhi);

    int64_t lo_ofs = lo - mylo, hi_ofs = hi - myhi;
    if (lo_ofs < 0  ||  hi_ofs > 0) {
        LOG_WARN(ga->g->glog, "[%d] garray access requested for invalid range"
                 " (%" PRId64 "-%" PRId64 ", have %" PRId64 "-%" PRId64 ")\n",
                 ga->g->rank, lo, hi, mylo, myhi);
        return -1;
    }

    if (ga->nlocal_elems > 0)
        *buf = &ga->buffer[lo_ofs * ga->elem_size];
    else
        LOG_INFO(ga->g->glog, "[%d] garray access requested, but no local"
                 " elements\n", ga->g->rank);

    return 0;
}


/*  garray_flush()
 */
void garray_flush(garray_t *ga)
{
    //MPI_Win_flush_all(ga->win);
}

