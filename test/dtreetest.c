/*  gasp -- global address space toolbox

    dtree -- distributed dynamic scheduler, microbenchmark
 */

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <sys/param.h>
#include <omp.h>
#include <pthread.h>
#include <math.h>
#include <mkl.h>
#include <immintrin.h>

#include "gasp.h"

#ifndef UINT64_MAX
#define UINT64_MAX      18446744073709551615ULL
#endif

#ifndef DBL_MAX
#define DBL_MAX         1.79769313486231470e+308
#endif

int num_ranks, my_rank;


double get_cpu_mhz()
{
    uint64_t t = rdtsc();
    sleep(1);
    return (rdtsc() - t) * 1.0 / 1e6;
}


void usage()
{
    if (my_rank == 0)
        fprintf(stderr, "Usage:\n"
                        "  dtreetest <#items> <mean> <stddev> \n"
                        "            <first> <rest> <mindist> \n"
                        "            <parents_work> [<fanout=1024>]\n");
}


int main(int argc, char **argv)
{
    int                 i, j, min_distrib, parents_work, fan_out = 1024,
                        is_parent;
    int64_t             num_work_items;
    double              mean, stddev, first, rest, cpu_mhz = get_cpu_mhz();
    dtree_t             *scheduler;
    int64_t volatile    lock __attribute((aligned(8)));
    gasp_t             *g;

    gasp_init(argc, argv, &g);
    num_ranks = gasp_nranks();
    my_rank = gasp_rank();

    if (argc < 8  ||  argc > 9) {
        usage();
        gasp_shutdown(g);
        exit(-1);
    }

    if (my_rank == 0) {
        for (i = 0;  i < argc;  i++)
            printf("%s ", argv[i]);
        printf("\nRoot CPU at %g MHz\n", cpu_mhz);
    }

    // get command line arguments
    num_work_items = strtol(argv[1], NULL, 10);
    mean = atof(argv[2]);
    stddev = atof(argv[3]);
    first = atof(argv[4]);
    rest = atof(argv[5]);
    min_distrib = strtol(argv[6], NULL, 10);
    parents_work = strtol(argv[7], NULL, 10);
    if (argc == 9)
        fan_out = strtol(argv[8], NULL, 10);

    div_t de = div(num_work_items, num_ranks);
    int64_t each = de.quot;
    if (de.rem > 0) ++each;

    if (my_rank == 0)
        printf("dtreetest:\n  %d ranks, ~%lu work items per rank\n",
               num_ranks, each);

#ifdef DEBUG_DTREE
    static int once = 0;
    if (!once) {
        once = 1;
        char *p = getenv("DEBUGWAIT");
        if (p) {
            int rank, wait_rank = strtol(p, NULL, 10);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == wait_rank) {
                int go = 0;
                while (!go)
                    cpupause();
            }
        }
    }
#endif  // DEBUG_DTREE

    if (my_rank == 0)
        printf("  generating random numbers...");

    // generate random numbers for work item durations--on each rank
    uint64_t *ticks = (uint64_t *)_mm_malloc(each * num_ranks * sizeof (uint64_t), 64);

    double *rnd, rm;
    rnd = (double *)_mm_malloc(each * sizeof (double), 64);

    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MRG32K3A, 7777777);

    int64_t cur = 0;
    for (i = 1;  i < num_ranks;  i++) {
        if (i % 2)
            vdRngUniform(0, stream, 1, &rm, mean-stddev, mean);
        else
            vdRngUniform(0, stream, 1, &rm, mean, mean+stddev);
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream,
                      each, rnd, rm, stddev);
        for (j = 0;  j < each;  j++)
            ticks[cur++] = (uint64_t)MAX(0, rnd[j]) * cpu_mhz * 1e6;
    }
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream,
                  each, rnd, mean, stddev);
    for (j = 0;  j < each;  j++)
        ticks[cur++] = (uint64_t)MAX(0, rnd[j]) * cpu_mhz * 1e6;

    vslDeleteStream(&stream);
    _mm_free(rnd);

    if (my_rank == 0)
        printf(" done.\n");

#ifdef DEBUG_DTREE
    char *m = getenv("UPPER");
    if (m) {
        int upper = strtol(m, NULL, 10);
        for (i = 0;  i < each;  i++) {
            if (ticks[i] > upper)
                printf("[%04d] ticks[%d] = %lu (was %g)\n", my_rank, i, ticks[i], (double)ticks[i]);
        }
    }
#endif

    // start OMP threads
    #pragma omp parallel
    {
        omp_get_thread_num();
    }

    lock_init(lock);

    int can_parent = 1;
    double rank_mul = 1.0;

#if (__MIC__)
    // TODO: this is okay for hetero, but wrong for MIC-only
    can_parent = 0;
    rank_mul = 6.0;
#endif

    if (my_rank == 0)
        printf("  creating tree...");

    dtree_create(g, fan_out, num_work_items, can_parent, parents_work,
            rank_mul, omp_get_max_threads(), omp_get_thread_num,
            first, rest, min_distrib, &scheduler, &is_parent);

    if (my_rank == 0)
        printf(" done.\n");

    // get initial allotment and start work
    int64_t volatile num_items;
    int64_t cur_item, last_item;

    num_items = dtree_initwork(scheduler, &cur_item, &last_item);
    int run_dtree = dtree_run(scheduler);

    uint64_t tinit_done = rdtsc();
    uint64_t wi_done = num_items;

    #pragma omp parallel
    {
        int tnum = omp_get_thread_num();

        // run the Dtree if needed
        if (run_dtree  &&  tnum == omp_get_num_threads() - 1) {
            DTREE_TRACE(scheduler, "[%04d] runner thread entering\n", my_rank);

            while (dtree_run(scheduler))
                cpupause();

            DTREE_TRACE(scheduler, "[%04d] runner thread exiting\n", my_rank);
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
                    //DTREE_TRACE(scheduler, "[%04d] getwork: %lld \n", my_rank, num_items);
                    continue;
                }
                item = cur_item++;
                lock_release(lock);

                waitcycles(ticks[item]);
            }
        }
    }

    uint64_t twork_done = rdtsc();
    double twork = ((double)(twork_done - tinit_done)) / (cpu_mhz * 1000);

    // wait for all ranks to be done
    MPI_Barrier(MPI_COMM_WORLD);

    uint64_t tall_done = rdtsc();
    double tall = ((double)(tall_done - tinit_done)) / (cpu_mhz * 1000),
           timbal = ((double)(tall_done - twork_done)) / (cpu_mhz * 1000);

    // clean up
    dtree_destroy(scheduler);

    double tmin_imbal, tmax_imbal, tall_imbal, tavg_imbal;
    MPI_Reduce(&timbal, &tmin_imbal, 1, MPI_DOUBLE, MPI_MIN,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&timbal, &tmax_imbal, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&timbal, &tall_imbal, 1, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);
    tavg_imbal = tall_imbal / num_ranks;

    // display timing info
    if (my_rank == 0) {
        double *tall_work = (double *)
                _mm_malloc(num_ranks * sizeof (double), 64);
        MPI_Gather(&twork, 1, MPI_DOUBLE, tall_work, 1, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        uint64_t *twi_done = (uint64_t *)
                _mm_malloc(num_ranks * sizeof (uint64_t), 64);
        MPI_Gather(&wi_done, 1, MPI_LONG, twi_done, 1, MPI_LONG,
                   0, MPI_COMM_WORLD);

        printf("---\nper-rank data:\n");

        wi_done = 0;
        double tmin = DBL_MAX, tmax = 0.0, ttotal_work = 0.0;
        for (i = 0;  i < num_ranks;  i++) {
            printf("  [%04d] work items: %lu; \ttime (msecs): %g)\n",
                   i, twi_done[i], tall_work[i]);
            if (tmin > tall_work[i]) tmin = tall_work[i];
            if (tmax < tall_work[i]) tmax = tall_work[i];
            ttotal_work += tall_work[i];
            wi_done += twi_done[i];
        }

        _mm_free(twi_done);
        _mm_free(tall_work);

        double best_avg = ttotal_work;

        printf("work items done: %lu\n---\n", wi_done);
        printf("runtime (msecs): %g\n", tall);
        printf("  avg runtime (bestcase) [min, max]: %g (%.2g%%) [%g, %g]\n",
               best_avg, (best_avg * 100) / tall, tmin, tmax);
        printf("  avg imbalance (msecs) [min, max]: %g (%.2g%%) [%g, %g]\n",
               tavg_imbal, (tavg_imbal * 100) / tall,
               tmin_imbal, tmax_imbal);
        printf("  avg work (msecs): %g (%.2g%%)\n",
               tall - tavg_imbal,
               ((tall - tavg_imbal) * 100) / tall);
    }
    else {
        MPI_Gather(&twork, 1, MPI_LONG, NULL, 0, MPI_LONG, 0, MPI_COMM_WORLD);
        MPI_Gather(&wi_done, 1, MPI_LONG, NULL, 0, MPI_LONG, 0, MPI_COMM_WORLD);
    }

    _mm_free(ticks);

    gasp_shutdown(g);

    return 0;
}

