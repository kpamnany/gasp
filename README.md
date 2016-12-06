gasp
====

[![Build Status](https://travis-ci.org/kpamnany/gasp.svg?branch=master)](https://travis-ci.org/kpamnany/gasp)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/kpamnany/gasp/blob/master/LICENSE)

Global Address SPace toolbox.

## Description

The [Dtree](http://dx.doi.org/10.1007/978-3-319-20119-1_10) distributed dynamic scheduler, and a fast minimal implementation global
arrays implementation with an interface based on [GA](http://hpc.pnl.gov/globalarrays/index.shtml).

+ uses MPI-2 asynchronous communication and MPI-3 one-sided RMA
+ runs on C11/Linux; tested on Cray machines and Intel clusters

## Usage

##### Dtree:

See the paper linked above for details on Dtree parameters. See `test/dtreetest.c` for an example.
```c
#include <omp.h>
#include "gasp.h"

gasp_t *g;
int64_t nid, nnodes;

int main(int argc, char **argv)
{
    /* initialize gasp */
    gasp_init(argc, argv, &g);
    nid = gasp_nodeid();
    nnodes = gasp_nnodes();

    /* required scheduler parameters */
    int64_t num_work_items = 50000;
    int min_distrib = omp_get_num_threads() * 1.5;
    double first = 0.4;

    /* parameters that usually don't need changing */
    int fan_out = 1024;
    int can_parent = 1;
    int parents_work = 1;
    double node_mul = 1.0;
    double rest = 0.4;

    /* create the scheduler */
    dtree_t *scheduler;
    int is_parent;
    dtree_create(g, fan_out, num_work_items, can_parent, parents_work,
            node_mul, omp_get_max_threads(), omp_get_thread_num,
            first, rest, min_distrib, &scheduler, &is_parent);

    /* get initial work allocation */
    int64_t volatile num_items;
    int64_t cur_item, last_item;

    num_items = dtree_initwork(scheduler, &cur_item, &last_item);
    int run_dtree = dtree_run(scheduler);

    int64_t volatile lock __attribute((aligned(8)));
    lock_init(lock);
    uint64_t wi_done = num_items;

    /* work loop */
    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();

        // run the Dtree if needed
        if (is_parent  &&  run_dtree  &&  tnum == omp_get_num_threads() - 1) {
            while (dtree_run(scheduler))
                cpu_pause();
        }

        else {
            int64_t item;

            while (num_items) {
                lock_acquire(lock);
                if (last_item == 0) {
                    lock_release(lock);
                    break;
                }
                if (cur_item == last_item) {
                    num_items = dtree_getwork(scheduler, &cur_item, &last_item);
                    wi_done += num_items;
                    lock_release(lock);
                    continue;
                }
                item = cur_item++;
                lock_release(lock);

                /* process `item` here */
            }
        }
    }

    /* clean up */
    dtree_destroy(scheduler);

    gasp_shutdown(g);
    return 0;
}
```

##### Global arrays:

See Global Arrays documentation for PGAS model and concepts. See `test/garraytest.c` for example.
```c
#include "gasp.h"

gasp_t *g;
int64_t nid, nnodes;

typedef struct aelem {
    int64_t a, b;
} aelem_t;

int main(int argc, char **argv)
{
    /* initialize gasp */
    gasp_init(argc, argv, &g);
    nid = gasp_nodeid();
    nnodes = gasp_nnodes();

    /* create a global array; currently only 1-dimensional arrays
       are supported, and chunks cannot be specified */
    garray_t *ga;
    int64_t nelems = nnodes * 100;
    int64_t dim[1] = { nelems };
    garray_create(g, 1, dim, sizeof(aelem_t), NULL, &ga);

    /* get the local part of the global array; lo-hi inclusive */
    int64_t lo[1], hi[1];
    garray_distribution(ga, nid, lo, hi);

    int64_t nlocal_elems = hi[0] - lo[0] + 1;

    aelem_t *aptr;
    garray_access(ga, lo, hi, (void **)&aptr);
    for (int64_t i = 0;  i < nlocal_elems;  ++i) {
        aptr[i].a = (i + 1) * nid;
        aptr[i].b = nid;
    }
    garray_flush(ga);

    /* wait for all nodes to complete */
    gasp_sync();

    /* put into the next node's first element */
    aelem_t tae;
    tae.a = 100 + nid;
    tae.b = nid;
    int64_t ti[1];
    ti[0] = (hi[0] + 1) % nelems;
    garray_put(ga, ti, ti, &tae);

    /* wait for all nodes to complete */
    gasp_sync();

    /* destroy the global array */
    garray_destroy(ga);

    gasp_shutdown(g);
    return 0;
}
```
