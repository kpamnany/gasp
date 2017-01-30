/*  gasp -- global address space toolbox

    toolbox initialization and utility functions
 */


#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <immintrin.h>

#include "gasp.h"


/* initialize and finalize MPI? */
static int mpi_was_initialized = 0;


/*  gasp_init()
 */
int64_t gasp_init(int ac, char **av, gasp_t **g_)
{
    MPI_Initialized(&mpi_was_initialized);

    if (!mpi_was_initialized) {
        int provided;
        MPI_Init_thread(&ac, &av, MPI_THREAD_MULTIPLE, &provided);
    }

    //gasp_t *g = aligned_alloc(64, sizeof(gasp_t));
    gasp_t *g;
    posix_memalign((void **)&g, 64, sizeof(gasp_t));
    g->nranks = gasp_nranks();
    g->rank = gasp_rank();
    log_init(&g->glog, "GARRAY_LOG_LEVEL");
    log_init(&g->dlog, "DTREE_LOG_LEVEL");
    *g_ = g;

    return 0;
}


/*  gasp_shutdown()
 */
void gasp_shutdown(gasp_t *g)
{
    free(g);
    if (!mpi_was_initialized)
        MPI_Finalize();
}


/*  gasp_nranks()
 */
int64_t gasp_nranks()
{
    int num_ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    return num_ranks;
}


/*  gasp_rank()
 */
int64_t gasp_rank()
{
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    return my_rank;
}


/*  gasp_sync_all()
 */
void gasp_sync()
{
    MPI_Barrier(MPI_COMM_WORLD);
}


/*  rdtsc()
 */
inline uint64_t __attribute__((always_inline)) rdtsc()
{
    uint32_t hi, lo;
    __asm__ __volatile__(
        "lfence\n\t"
        "rdtsc"
        : "=a"(lo), "=d"(hi)
        : /* no inputs */
        : "rbx", "rcx");
    return ((uint64_t)hi << 32ull) | (uint64_t)lo;
}


/*  cpu_pause()
 */
inline void __attribute__((always_inline)) cpu_pause()
{
    _mm_pause();
}


inline void __attribute__((always_inline)) start_sde_tracing()
{
    __SSC_MARK(0x111);
}


inline void __attribute__((always_inline)) stop_sde_tracing()
{
    __SSC_MARK(0x222);
}

