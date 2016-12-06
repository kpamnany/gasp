/*  gasp -- global address space toolbox

    global arrays tests
 */

#include <stdio.h>
#include <stdlib.h>
#include "gasp.h"


typedef struct aelem_tag {
    int64_t idx;
    int64_t rank;
} aelem_t;


gasp_t *g;
int64_t nid, nnodes;


void check_distrib_access_get(garray_t *ga);
void check_put_get(garray_t *ga);


int main(int argc, char **argv)
{
    /* initialize gasp */
    gasp_init(argc, argv, &g);
    nid = gasp_nodeid();
    nnodes = gasp_nnodes();
    if (nid == 0)
        printf("gasptest -- %lld nodes\n", nnodes);

    /* even distribution global array tests */
    int64_t nelems = nnodes * 5;
    garray_t *ga;
    int64_t dim1[1] = { nelems };
    garray_create(g, 1, dim1, sizeof (aelem_t), NULL, &ga);

    if (nid == 0)
        printf("[%lld] even distribution global array tests\n", nid);
    check_distrib_access_get(ga);
    check_put_get(ga);

    garray_destroy(ga);

    /* uneven distribution global array tests */
    nelems = nelems + (nnodes / 2);
    dim1[0] = nelems;
    garray_create(g, 1, dim1, sizeof (aelem_t), NULL, &ga);

    if (nid == 0)
        printf("[%lld] uneven distribution global array tests\n", nid);
    check_distrib_access_get(ga);
    check_put_get(ga);

    garray_destroy(ga);

    gasp_shutdown(g);

    return 0;
}


void check_distrib_access_get(garray_t *ga)
{
    /* get the local part of the global array and write into it */
    int64_t lo[1], hi[1];
    garray_distribution(ga, nid, lo, hi);

    int64_t nlocal_elems = hi[0] - lo[0] + 1;

    aelem_t *aptr;
    garray_access(ga, lo, hi, (void **)&aptr);
    for (int64_t i = 0;  i < nlocal_elems;  ++i) {
        aptr[i].idx = (i + 1) * nid;
        aptr[i].rank = nid;
    }
    garray_flush(ga);

    /* wait for all nodes to complete */
    gasp_sync();

    /* get the whole array on node 0 and check it */
    if (nid == 0) {
        int64_t nelems = garray_length(ga);
        aelem_t *arr = (aelem_t *)malloc(nelems * sizeof (aelem_t));
        int64_t flo[1], fhi[1];
        flo[0] = 0;
        fhi[0] = nelems - 1;
        garray_get(ga, flo, fhi, arr);

        ldiv_t res = ldiv(nelems, nnodes);
        int passed = 1;
        for (int64_t n = 0;  n < nnodes;  ++n) {
            int64_t lidx = (n * res.quot) + (n < res.rem ? n : res.rem);
            if (arr[lidx].rank != n) {
                printf("array[%lld] != %lld\n", lidx, n);
                passed = 0;
            }
        }
        printf("distribution/access/get tests %s\n", passed ? "passed" : "failed");
        free(arr);
    }

    gasp_sync();
}


void check_put_get(garray_t *ga)
{
    int64_t nelems = garray_length(ga);

    int64_t lo[1], hi[1];
    garray_distribution(ga, nid, lo, hi);

    /* test put by writing to the next node's first element */
    aelem_t tae;
    tae.idx = 100 + nid;
    tae.rank = nid;
    int64_t ti[1];
    ti[0] = (hi[0] + 1) % nelems;
    garray_put(ga, ti, ti, &tae);

    /* wait for all nodes to complete */
    gasp_sync();

    /* get the whole array on node 0 and check it */
    if (nid == 0) {
        aelem_t *arr = (aelem_t *)malloc(nelems * sizeof (aelem_t));
        lo[0] = 0;
        hi[0] = nelems - 1;
        garray_get(ga, lo, hi, arr);

        ldiv_t res = ldiv(nelems, nnodes);
        int passed = 1;
        int64_t srcn = nnodes - 1;
        for (int64_t n = 0;  n < nnodes;  ++n) {
            int64_t lidx = (n * res.quot) + (n < res.rem ? n : res.rem);
            if (arr[lidx].idx != 100 + srcn) {
                printf("array[%lld].idx != %lld]\n", lidx, 100 + srcn);
                passed = 0;
            }
            srcn = (srcn + 1) % nnodes;
        }
        printf("put/get tests %s\n", passed ? "passed" : "failed");
        free(arr);
    }

    gasp_sync();
}

