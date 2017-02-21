/*  gasp -- global address space toolbox

    global arrays tests
 */

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gasp.h"


typedef struct aelem_tag {
    int64_t idx;
    int64_t rank;
} aelem_t;


gasp_t *g;
int rank, nranks;


void check_distrib_access_get(garray_t *ga);
void check_put_get(garray_t *ga);


int main(int argc, char **argv)
{
    /* initialize gasp */
    gasp_init(argc, argv, &g);
    rank = gasp_rank();
    nranks = gasp_nranks();
    if (rank == 0)
        printf("gasptest -- %d ranks\n", nranks);

    /* even distribution global array tests */
    int64_t nelems = nranks * 5;
    garray_t *ga;
    garray_create(g, nelems, sizeof (aelem_t), NULL, &ga);

    if (rank == 0)
        printf("[%d] even distribution global array tests\n", rank);
    check_distrib_access_get(ga);
    check_put_get(ga);

    garray_destroy(ga);

    /* uneven distribution global array tests */
    nelems = nelems + (nranks / 2);
    garray_create(g, nelems, sizeof (aelem_t), NULL, &ga);

    if (rank == 0)
        printf("[%d] uneven distribution global array tests\n", rank);
    check_distrib_access_get(ga);
    check_put_get(ga);

    garray_destroy(ga);

    gasp_shutdown(g);

    return 0;
}


void check_distrib_access_get(garray_t *ga)
{
    /* get the local part of the global array and write into it */
    int64_t lo, hi;
    garray_distribution(ga, rank, &lo, &hi);

    int64_t nlocal_elems = hi - lo + 1;

    aelem_t *aptr;
    garray_access(ga, lo, hi, (void **)&aptr);
    for (int64_t i = 0;  i < nlocal_elems;  ++i) {
        aptr[i].idx = (i + 1) * rank;
        aptr[i].rank = rank;
    }
    garray_flush(ga);

    /* wait for all ranks to complete */
    gasp_sync();

    /* get the whole array on rank 0 and check it */
    if (rank == 0) {
        int64_t nelems = garray_length(ga);
        aelem_t *arr = (aelem_t *)malloc(nelems * sizeof (aelem_t));
        int64_t flo, fhi;
        flo = 0;
        fhi = nelems - 1;
        garray_get(ga, flo, fhi, arr);

        ldiv_t res = ldiv(nelems, nranks);
        int passed = 1;
        for (int64_t n = 0;  n < nranks;  ++n) {
            int64_t lidx = (n * res.quot) + (n < res.rem ? n : res.rem);
            if (arr[lidx].rank != n) {
                printf("array[%" PRId64 "] != %" PRId64 "\n", lidx, n);
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

    int64_t lo, hi;
    garray_distribution(ga, rank, &lo, &hi);

    /* test put by writing to the next rank's first element */
    aelem_t tae;
    tae.idx = 100 + rank;
    tae.rank = rank;
    int64_t ti;
    ti = (hi + 1) % nelems;
    garray_put(ga, ti, ti, &tae);

    /* wait for all ranks to complete */
    gasp_sync();

    /* get the whole array on rank 0 and check it */
    if (rank == 0) {
        aelem_t *arr = (aelem_t *)malloc(nelems * sizeof (aelem_t));
        lo = 0;
        hi = nelems - 1;
        garray_get(ga, lo, hi, arr);

        ldiv_t res = ldiv(nelems, nranks);
        int passed = 1;
        int64_t srcn = nranks - 1;
        for (int64_t n = 0;  n < nranks;  ++n) {
            int64_t lidx = (n * res.quot) + (n < res.rem ? n : res.rem);
            if (arr[lidx].idx != 100 + srcn) {
                printf("array[%" PRId64 "].idx != %" PRId64 "]\n", lidx, 100 + srcn);
                passed = 0;
            }
            srcn = (srcn + 1) % nranks;
        }
        printf("put/get tests %s\n", passed ? "passed" : "failed");
        free(arr);
    }

    gasp_sync();
}

