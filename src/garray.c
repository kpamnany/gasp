/*  gasp -- global address space toolbox

    global arrays implementation
*/


#include <mpi.h>
#include <assert.h>
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
int64_t garray_create(gasp_t *g, int64_t ndims, int64_t *dims, int64_t elem_size,
            int64_t *chunks, garray_t **ga_)
{
    /* only 1D for now */
    assert(ndims == 1);

    if (ndims < 1  ||  ndims > GARRAY_MAX_DIMS)
        return -1;

    /* only regular distribution for now */
    assert(chunks == NULL);

    garray_t *ga;
    posix_memalign((void **)&ga, 64, sizeof(garray_t));

    ga->g = g;

    /* distribution of elements */
    ldiv_t res = ldiv(dims[0], g->nnodes);
    ga->nextra_elems = res.rem;
    ga->nelems_per_node = ga->nlocal_elems = res.quot;
    if (g->nid < ga->nextra_elems)
        ++ga->nlocal_elems;

    /* fill in array info */
    ga->ndims = ndims;
    ga->dims = (int64_t *)malloc(ndims * sizeof(int64_t));
    for (int64_t i = 0;  i < ndims;  ++i)
        ga->dims[i] = dims[i];
    ga->elem_size = elem_size;

    /* allocate the array */
    MPI_Win_allocate(ga->nlocal_elems*elem_size, 1, MPI_INFO_NULL,
            MPI_COMM_WORLD, &ga->buffer, &ga->win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, ga->win);
    memset(ga->buffer, 0, ga->nlocal_elems*elem_size);
    MPI_Barrier(MPI_COMM_WORLD);

    *ga_ = ga;

    LOG_INFO(g->glog, "[%d] garray created %lld-array, element size %lld\n",
             g->nid, ndims, elem_size);

    return 0;
}


/*  garray_destroy()
 */
void garray_destroy(garray_t *ga)
{
    garray_flush(ga);
    MPI_Win_unlock_all(ga->win);
    MPI_Win_free(&ga->win);
    free(ga->dims);

    LOG_INFO(ga->g->glog, "[%d] garray destroyed %lld-array, element size %lld\n",
             ga->g->nid, ga->ndims, ga->elem_size);

    free(ga);
}


/*  garray_ndims()
 */
int64_t garray_ndims(garray_t *ga)
{
    return ga->ndims;
}


/*  garray_length()
 */
int64_t garray_length(garray_t *ga)
{
    int64_t r = 1;

    for (int64_t i = 0;  i < ga->ndims;  ++i)
        r *= ga->dims[i];

    return r;
}


/*  garary_size()
 */
int64_t garray_size(garray_t *ga, int64_t *dims)
{
    for (int64_t i = 0;  i < ga->ndims;  ++i)
        dims[i] = ga->dims[i];

    return 0;
}


/*  calc_target()
 */
static void calc_target(garray_t *ga, int64_t gidx, int64_t *tnid_, int64_t *tidx_)
{
    /* if this node has no local elements, there are less than nnodes
       elements, so the array index is the node index */
    if (ga->nlocal_elems == 0) {
        *tnid_ = gidx;
        *tidx_ = 0;
        return;
    }

    /* compute the target nid+idx */
    ldiv_t res = ldiv(gidx, ga->nlocal_elems);

    /* if the distribution is not perfectly even, we have to adjust
       the target nid+idx appropriately */
    if (ga->nextra_elems > 0) {
        int64_t tnid = res.quot, tidx = res.rem;

        /* if i have an extra element... */
        if (ga->g->nid < ga->nextra_elems) {
            /* but the target does not */
            if (tnid >= ga->nextra_elems) {
                /* then the target index has to be adjusted upwards */
                tidx += (tnid - ga->nextra_elems);
                /* which may mean that the target nid does too */
                while (tidx >= (ga->nlocal_elems - 1)) {
                    ++tnid;
                    tidx -= (ga->nlocal_elems - 1);
                }
            }
        }

        /* i don't have an extra element... */
        else {
            /* so adjust the target index downwards */
            tidx -= (tnid < ga->nextra_elems ? tnid : ga->nextra_elems);
            /* which may mean the target nid has to be adjusted too */
            while (tidx < 0) {
                --tnid;
                tidx += ga->nelems_per_node + (tnid < ga->nextra_elems ? 1 : 0);
            }
        }

        res.quot = tnid; res.rem = tidx;
    }

    LOG_DEBUG(ga->g->glog, "[%d] garray calc %lld, target %ld.%ld\n",
              ga->g->nid, gidx, res.quot, res.rem);

    *tnid_ = res.quot;
    *tidx_ = res.rem;
}


/*  garray_get()
 */
int64_t garray_get(garray_t *ga, int64_t *lo, int64_t *hi, void *buf_)
{
    int64_t count = (hi[0] - lo[0]) + 1, length = count * ga->elem_size,
            tlonid, tloidx, thinid, thiidx, tnid, tidx, n, oidx = 0;
    int8_t *buf = (int8_t *)buf_;

    calc_target(ga, lo[0], &tlonid, &tloidx);
    calc_target(ga, hi[0], &thinid, &thiidx);

    /* is all requested data on the same target? */
    if (tlonid == thinid) {
        LOG_DEBUG(ga->g->glog, "[%d] garray getting %lld-%lld, single target %lld.%lld\n",
                  ga->g->nid, lo[0], hi[0], tlonid, tloidx);

        //MPI_Win_lock(MPI_LOCK_SHARED, tlonid, 0, ga->win);
        MPI_Get(buf, length, MPI_INT8_T,
                tlonid, (tloidx * ga->elem_size), length, MPI_INT8_T,
                ga->win);
        //MPI_Win_unlock(tlonid, ga->win);
        MPI_Win_flush_local(tlonid, ga->win);

        return 0;
    }

    /* get the data in the lo nid */
    n = ga->nelems_per_node + (tlonid < ga->nextra_elems ? 1 : 0) - tloidx;

    LOG_DEBUG(ga->g->glog, "[%d] garray getting %lld elements from %lld.%lld\n",
              ga->g->nid, n, tlonid, tloidx);

    //MPI_Win_lock(MPI_LOCK_SHARED, tlonid, 0, ga->win);
    MPI_Get(buf, (n * ga->elem_size), MPI_INT8_T,
            tlonid, (tloidx * ga->elem_size), (n * ga->elem_size), MPI_INT8_T,
            ga->win);
    //MPI_Win_unlock(tlonid, ga->win);

    oidx = (n * ga->elem_size);

    /* get the data in the in-between nids */
    tidx = 0;
    for (tnid = tlonid + 1;  tnid < thinid;  ++tnid) {
        n = ga->nelems_per_node + (tnid < ga->nextra_elems ? 1 : 0);

        LOG_DEBUG(ga->g->glog, "[%d] garray getting %lld elements from %lld.%lld\n",
                  ga->g->nid, n, tnid, tidx);

        //MPI_Win_lock(MPI_LOCK_SHARED, tnid, 0, ga->win);
        MPI_Get(&buf[oidx], (n * ga->elem_size), MPI_INT8_T,
                tnid, 0, (n * ga->elem_size), MPI_INT8_T,
                ga->win);
        //MPI_Win_unlock(tnid, ga->win);

        oidx += (n * ga->elem_size);
    }

    /* get the data in the hi nid */
    n = thiidx + 1;

    LOG_DEBUG(ga->g->glog, "[%d] garray getting %lld elements up to %lld.%lld\n",
              ga->g->nid, n, thinid, thiidx);

    //MPI_Win_lock(MPI_LOCK_SHARED, thinid, 0, ga->win);
    MPI_Get(&buf[oidx], (n * ga->elem_size), MPI_INT8_T,
            thinid, 0, (n * ga->elem_size), MPI_INT8_T,
            ga->win);
    //MPI_Win_unlock(thinid, ga->win);
    MPI_Win_flush_local_all(ga->win);

    return 0;
}


/*  garray_put()
 */
int64_t garray_put(garray_t *ga, int64_t *lo, int64_t *hi, void *buf_)
{
    int64_t count = (hi[0] - lo[0]) + 1, length = count * ga->elem_size,
            tlonid, tloidx, thinid, thiidx, tnid, tidx, n, oidx = 0;
    int8_t *buf = (int8_t *)buf_;

    calc_target(ga, lo[0], &tlonid, &tloidx);
    calc_target(ga, hi[0], &thinid, &thiidx);

    /* is all data going to the same target? */
    if (tlonid == thinid) {
        LOG_DEBUG(ga->g->glog, "[%d] garray put %lld-%lld, single target %lld.%lld\n",
                  ga->g->nid, lo[0], hi[0], tlonid, tloidx);

        //MPI_Win_lock(MPI_LOCK_EXCLUSIVE, tlonid, 0, ga->win);
        MPI_Put(buf, length, MPI_INT8_T,
                tlonid, (tloidx * ga->elem_size), length, MPI_INT8_T,
                ga->win);
        //MPI_Win_unlock(tlonid, ga->win);
        MPI_Win_flush(tlonid, ga->win);

        return 0;
    }

    /* put the data into the lo nid */
    n = ga->nelems_per_node + (tlonid < ga->nextra_elems ? 1 : 0) - tloidx;

    LOG_DEBUG(ga->g->glog, "[%d] garray putting %lld elements into %lld.%lld\n",
              ga->g->nid, n, tlonid, tloidx);

    //MPI_Win_lock(MPI_LOCK_SHARED, tlonid, 0, ga->win);
    MPI_Put(buf, length, MPI_INT8_T,
            tlonid, (tloidx * ga->elem_size), (n * ga->elem_size), MPI_INT8_T,
            ga->win);
    //MPI_Win_unlock(tlonid, ga->win);

    oidx = (n*ga->elem_size);

    /* put the data into the in-between nids */
    tidx = 0;
    for (tnid = tlonid + 1;  tnid < thinid;  ++tnid) {
        n = ga->nelems_per_node + (tnid < ga->nextra_elems ? 1 : 0);

        LOG_DEBUG(ga->g->glog, "[%d] garray putting %lld elements into %lld.%lld\n",
                  ga->g->nid, n, tnid, tidx);

        //MPI_Win_lock(MPI_LOCK_EXCLUSIVE, tnid, 0, ga->win);
        MPI_Put(&buf[oidx], (n * ga->elem_size), MPI_INT8_T,
                tnid, 0, (n * ga->elem_size), MPI_INT8_T,
                ga->win);
        //MPI_Win_unlock(tnid, ga->win);

        oidx += (n*ga->elem_size);
    }

    /* put the data into the hi nid */
    n = thiidx + 1;

    LOG_DEBUG(ga->g->glog, "[%d] garray putting %lld elements up to %lld.%lld\n",
              ga->g->nid, n, thinid, thiidx);

    //MPI_Win_lock(MPI_LOCK_EXCLUSIVE, thinid, 0, ga->win);
    MPI_Put(&buf[oidx], (n * ga->elem_size), MPI_INT8_T,
            thinid, 0, (n * ga->elem_size), MPI_INT8_T,
            ga->win);
    //MPI_Win_unlock(thinid, ga->win);
    MPI_Win_flush_all(ga->win);

    return 0;
}


/*  garray_distribution()
 */
int64_t garray_distribution(garray_t *ga, int64_t nid, int64_t *lo, int64_t *hi)
{
    if (nid >= ga->g->nnodes) {
        LOG_WARN(ga->g->glog, "[%d] garray distribution requested for node %lld out"
                  " of %d nodes\n", ga->g->nid, nid, ga->g->nnodes);
        return -1;
    }

    int64_t a1, a2;
    if (nid < ga->nextra_elems) {
        a1 = nid;
        a2 = 1;
    }
    else {
        a1 = ga->nextra_elems;
        a2 = 0;
    }
    lo[0] = nid * ga->nelems_per_node + a1;
    hi[0] = lo[0] + ga->nelems_per_node + a2 - 1; /* inclusive */

    return 0;
}


/*  garray_access()
 */
int64_t garray_access(garray_t *ga, int64_t *lo, int64_t *hi, void **buf)
{
    *buf = NULL;

    int64_t mylo[GARRAY_MAX_DIMS], myhi[GARRAY_MAX_DIMS];
    garray_distribution(ga, ga->g->nid, mylo, myhi);

    int64_t lo_ofs = lo[0]-mylo[0], hi_ofs = hi[0]-myhi[0];
    if (lo_ofs < 0  ||  hi_ofs > 0) {
        LOG_WARN(ga->g->glog, "[%d] garray access requested for invalid range"
                 " (%lld-%lld, have %lld-%lld)\n", ga->g->nid, lo[0], hi[0],
                 mylo[0], myhi[0]);
        return -1;
    }

    if (ga->nlocal_elems > 0)
        *buf = &ga->buffer[lo_ofs*ga->elem_size];
    else
        LOG_INFO(ga->g->glog, "[%d] garray access requested, but no local"
                 " elements\n", ga->g->nid);

    return 0;
}


/*  garray_flush()
 */
void garray_flush(garray_t *ga)
{
    //MPI_Win_flush_all(ga->win);
}

