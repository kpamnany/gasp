/*  gasp -- global address space toolbox

    dtree -- distributed dynamic scheduler, implementation
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


#ifndef UINT64_MAX
#define UINT64_MAX      18446744073709551615ULL
#endif


#ifndef SHOW_DTREE
#ifdef DEBUG_DTREE
#define SHOW_DTREE      1
#else
#define SHOW_DTREE      0
#endif
#endif


#ifdef PROFILE_DTREE

#define PROFILE_START(dt,w)                                                             \
    (dt)->times[(*(dt)->threadid)()][(w)].last = _rdtsc()

#define PROFILE_STAMP(dt,w)                                                             \
{                                                                                       \
    int t = (*(dt)->threadid)();                                                        \
    uint64_t l = (dt)->times[t][(w)].last = _rdtsc() - (dt)->times[t][(w)].last;        \
    if (l < (dt)->times[t][(w)].min) (dt)->times[t][(w)].min = l;                       \
    if (l > (dt)->times[t][(w)].max) (dt)->times[t][(w)].max = l;                       \
    (dt)->times[t][(w)].total += l;                                                     \
    ++(dt)->times[t][(w)].count;                                                        \
}

#else  /* !PROFILE_DTREE */

#define PROFILE_START(dt,w)
#define PROFILE_STAMP(dt,w)

#endif /* PROFILE_DTREE */


#ifdef PROFILE_DTREE

/*  parse_csv_str() -- parses a string of comma-separated numbers and returns
        an array containing the numbers in *vals, and the number of elements
        of the array. *vals must be freed by the caller!
 */
static int parse_csv_str(char *s, int **vals)
{
    int i, num = 0, *v;
    char *cp, *np;

    *vals = NULL;
    if (!s) return 0;

    cp = s;
    while (cp) {
        ++num;
        cp = strchr(cp, ',');
        if (cp) ++cp;
    }

    v = (int *)calloc(num, sizeof (int));

    cp = s;
    i = 0;
    while (cp) {
        np = strchr(cp, ',');
        if (np) *np++ = '\0';
        v[i++] = (int)strtol(cp, NULL, 10);
        cp = np;
    }

    *vals = v;
    return num;
}

#endif /* PROFILE_DTREE */


/*  build_tree() -- builds a tree of ranks with height either 2 or 3,
      depending on fan_out. The maximum number of ranks possible are leaf
      ranks.
 */
static void build_tree(dtree_t *dt, int fan_out, int can_parent)
{
    int i;

    /* almost all the ranks are leaf ranks */
    dt->num_children = 0;
    dt->children = NULL;

    /* compute how many parent ranks are needed */
    int num_parents = 0;
    div_t result = div(dt->g->nranks, fan_out);
    num_parents += result.quot;
    if (result.rem >= fan_out / 2) ++num_parents;
    if (num_parents > 1) ++num_parents;
    num_parents = MAX(1, num_parents);

    /* if there's just one parent rank, all other ranks are children --
        the tree height is 2
     */
    if (num_parents == 1) {
        dt->num_levels = 2;
        if (dt->g->rank == 0) {
            dt->my_level = 0;
            dt->num_children = dt->g->nranks - 1;
            dt->children = (int *)_mm_malloc(dt->num_children * sizeof (int), 64);
            for (i = 0;  i < dt->num_children;  ++i)
                dt->children[i] = i + 1;
            dt->parent = -1;
        }
        else {
            dt->my_level = 1;
            dt->parent = 0;
        }

        return;
    }

    /* more than one parent rank: all the root rank's children are
       parents--the tree height is 3
     */
    dt->num_levels = 3;
    int *rank_can_parent = (int *)calloc(dt->g->nranks, sizeof (int));
    MPI_Allgather(&can_parent, 1, MPI_INT, rank_can_parent, 1, MPI_INT,
                  MPI_COMM_WORLD);

    /* the root rank identifies its children */
    if (dt->g->rank == 0) {
        dt->my_level = 0;
        dt->num_children = MAX(1, num_parents - 1);
        dt->children = (int *)_mm_malloc(dt->num_children * sizeof (int), 64);
        int my_child = 0;
        for (i = 1;  i < dt->g->nranks;  ++i) {
            if (rank_can_parent[i]) {
                dt->children[my_child++] = i;
                if (my_child == dt->num_children)
                    break;
            }
        }
        assert(my_child == dt->num_children);
        dt->parent = -1;
    }

    /* other ranks figure out their parents (and maybe children) */
    else {
        /* count the number of parents before me */
        int pcount = 0;
        for (i = dt->g->rank - 1;  i >= 0;  --i) {
            if (rank_can_parent[i]) {
                ++pcount;
                if (pcount == num_parents)
                    break;
            }
        }

        /* the number of children before me */
        int ccount = dt->g->rank - pcount;

        /* balance children among parents */
        result = div((dt->g->nranks - num_parents), (num_parents - 1));

        /* parents figure out their children */
        if (can_parent  &&  pcount < num_parents) {
            dt->my_level = 1;
            int i_parent = pcount - 1;
            int first = i_parent * result.quot;
            int last = first + result.quot;
            if (result.rem > 0) {
                if (i_parent < result.rem) {
                    first += i_parent;
                    last += i_parent + 1;
                }
                else {
                    first += result.rem;
                    last += result.rem;
                }
            }

            dt->num_children = last - first;
            dt->children = (int *)_mm_malloc(dt->num_children * sizeof (int), 64);

            /* count out my children */
            int cur_child = 0, my_child = 0, parents_seen = 0;
            for (i = 0;  i < dt->g->nranks;  ++i) {
                if (parents_seen < num_parents) {
                    if (rank_can_parent[i]) {
                        ++parents_seen;
                        continue;
                    }
                }
                if (cur_child++ < first)
                    continue;
                dt->children[my_child++] = i;
                if (my_child == dt->num_children)
                    break;
            }
            assert(my_child == dt->num_children);

            /* parent is the root */
            dt->parent = 0;
        }

        /* children need to figure out their parent */
        else {
            dt->my_level = 2;
            dt->parent = -1;
            int d = result.quot;
            if (result.rem > 0) ++d;
            int parent_idx = (ccount / d) + 1;
            int p = 0;
            for (i = 0;  i < dt->g->nranks;  ++i) {
                if (rank_can_parent[i]) {
                    if (p++ == parent_idx) {
                        dt->parent = i;
                        break;
                    }
                }
            }
            assert(dt->parent != -1);
        }
    }
    free(rank_can_parent);
}


/*  init_distrib_fractions() -- initialize distribution fractions based
        on the sizes of children's sub-trees. These are fractional
        sizes, based on a child's relative compute capacity.
 */
static void init_distrib_fractions(dtree_t *dt)
{
    int i;

    dt->distrib_fractions[0] = 0.0;
    dt->tot_children = 0.0;

    /* leaf ranks start the reporting */
    if (dt->num_children == 0) {
        /* at leaf ranks, we give all the work */
        dt->distrib_fractions[0] = 1.0;

        /* report rank multiplier to parent */
        if (dt->g->rank != 0)
            MPI_Send(&dt->rank_mul, 1, MPI_DOUBLE, dt->parent, 0, MPI_COMM_WORLD);
    }

    /* reports work their way up */
    else {
        double *children_sizes;

        children_sizes = (double *)calloc(dt->num_children, sizeof (double));

        /* post receives for children to report sub-tree sizes */
        for (i = 0;  i < dt->num_children;  i++)
            MPI_Irecv(&children_sizes[i], 1, MPI_DOUBLE, dt->children[i], 0,
                      MPI_COMM_WORLD, &dt->children_reqs[i]);

        MPI_Waitall(dt->num_children, dt->children_reqs, MPI_STATUSES_IGNORE);

        /* count total children */
        for (i = 0;  i < dt->num_children;  i++)
            dt->tot_children += children_sizes[i];

        /* add me as a child too, only if parents work */
        if (dt->parents_work)
            dt->tot_children += dt->rank_mul;

        /* report this up if I have a parent */
        if (dt->parent != -1)
            MPI_Send(&dt->tot_children, 1, MPI_DOUBLE, dt->parent, 0, MPI_COMM_WORLD);

        /* calculate the fractions for each child (again, including myself) */
        dt->distrib_fractions[0] = dt->parents_work ? dt->rank_mul / dt->tot_children : 0.0;
        for (i = 0;  i < dt->num_children;  i++)
            dt->distrib_fractions[i + 1] = children_sizes[i] / dt->tot_children;

        free(children_sizes);
    }
}


/*  dtree_create()
 */
int dtree_create(gasp_t *g,
        int fan_out_,
        int64_t num_work_items_,
        int can_parent_,
        int parents_work_,
        double rank_mul_,
        int num_threads_,
        int (*threadid_)(),
        double first_,
        double rest_,
        int16_t min_distrib_,
        dtree_t **dt_,
        int *is_parent_)
{
    *dt_ = NULL;

    /* sanity checks */
    assert(first_ > 0.0  &&  first_ <= 1.0  &&  rest_ > 0.0  &&  rest_ <= 1.0);
    assert(min_distrib_ >= 1);

    dtree_t *dt = (dtree_t *)_mm_malloc(sizeof (dtree_t), 64);
    if (dt == NULL)
        return -1;

    dt->last_work_item = num_work_items_;
    dt->parents_work = (int16_t)parents_work_;
    dt->rank_mul = rank_mul_;
    dt->num_threads = num_threads_;
    dt->threadid = threadid_;
    dt->first = first_;
    dt->rest = rest_;
    dt->min_distrib = min_distrib_ * rank_mul_;

    dt->first_work_item = dt->next_work_item = 0;

    lock_init(dt->work_lock);

    dt->g = g;

    if (dt->g->rank == 0)
        assert(can_parent_);

    build_tree(dt, fan_out_, can_parent_);

    /* leaf ranks don't scale with `rest` */
    if (dt->num_children == 0)
        dt->rest = 1.0;

    /* distribution fractions */
    dt->distrib_fractions = (double *)
            _mm_malloc((dt->num_children + 1) * sizeof (double), 64);

#if SHOW_DTREE
    if (dt->num_children) {
        printf("[%04d] parent=[%04d], #children=%d (",
               dt->g->rank, dt->parent, dt->num_children);
        for (int i = 0;  i < dt->num_children-1;  i++)
            printf("%d,", dt->children[i]);
        printf("%d)\n", dt->children[dt->num_children-1]);
    }
#endif
#if DEBUG_DTREE
    printf("[%04d] first wi=%d, last wi=%d, first=%f, rest=%f\n",
            dt->g->rank, dt->first_work_item, dt->last_work_item,
            dt->first, dt->rest);
#endif

    /* create MPI receive buffers and request handles for children */
    dt->children_req_bufs = NULL;
    if (dt->num_children > 0) {
        dt->children_req_bufs = (int16_t *)
                _mm_malloc(dt->num_children * sizeof (int16_t), 64);
        dt->children_reqs = (MPI_Request *)
                _mm_malloc(dt->num_children * sizeof (MPI_Request), 64);
    }

#ifdef PROFILE_DTREE
    dt->times = (thread_timing_t **)
            _mm_malloc(dt->num_threads * sizeof (thread_timing_t *), 64);
    int tnum;
    for (tnum = 0;  tnum < dt->num_threads;  tnum++) {
        dt->times[tnum] = (thread_timing_t *)
                _mm_malloc(NTIMES * sizeof (thread_timing_t), 64);
        for (i = 0;  i < NTIMES;  i++) {
            dt->times[tnum][i].last = dt->times[tnum][i].max = 
                dt->times[tnum][i].total = dt->times[tnum][i].count = 0;
            dt->times[tnum][i].min = UINT64_MAX;
        }
    }
#endif

    *dt_ = dt;
    *is_parent_ = (dt->num_children > 0);
    return 0;
}


/*  dtree_destroy()
 */
void dtree_destroy(dtree_t *dt)
{
    DTREE_TRACE(dt, "[%04d] fini\n", dt->g->rank);
    MPI_Barrier(MPI_COMM_WORLD);

#ifdef PROFILE_DTREE
    // get CPU speed
    uint64_t t = _rdtsc();
    sleep(1);
    double cpu_mhz = (_rdtsc() - t) * 1.0 / 1e6;

    // reduce all threads' timing data
    thread_timing_t coll_times[NTIMES];
    for (int i = 0;  i < NTIMES;  i++) {
        memset(&coll_times[i], 0, sizeof (thread_timing_t));
        coll_times[i].min = UINT64_MAX;
    }
    for (int tnum = 0;  tnum < dt->num_threads;  tnum++) {
        for (int i = 0;  i < NTIMES;  i++) {
            coll_times[i].total += dt->times[tnum][i].total;
            coll_times[i].count += dt->times[tnum][i].count;
            if (dt->times[tnum][i].max > coll_times[i].max)
                coll_times[i].max = dt->times[tnum][i].max;
            if (dt->times[tnum][i].min < coll_times[i].min)
                coll_times[i].min = dt->times[tnum][i].min;
        }
    }

    // specified ranks should dump individual profiling data
    int num_pranks, *pranks;
    char *cp = getenv("DTREE_PRANKS");
    num_pranks = parse_csv_str(cp, &pranks);
    for (int j = 0;  j < num_pranks;  j++) {
        if (dt->g->rank == pranks[j]) {
            printf("[%04d] Dtree profile: #calls, total (usecs), "
                   "avg [min, max] (usecs)\n",
                   dt->g->rank);
            for (int i = 0;  i < NTIMES;  i++) {
                double a = 0, t = coll_times[i].total / cpu_mhz;
                if (coll_times[i].count > 0)
                    a = t / (double)coll_times[i].count;
                printf("[%04d] %s: %llu, %g, %g, [%g, %g]\n", dt->g->rank,
                       times_names[i], coll_times[i].count, t, a,
                       coll_times[i].min / cpu_mhz, 
                       coll_times[i].max / cpu_mhz);
            }
            break;
        }
    }
    free(pranks);

    // reduce MPI ranks' timing data, averaging by level
    uint64_t *min[NTIMES], *max[NTIMES], *count[NTIMES], *avg[NTIMES],
             *ranks[NTIMES], u, r, a;

    for (int i = 0;  i < NTIMES;  i++) {
        min[i] = (uint64_t *)calloc(dt->num_levels, sizeof (uint64_t));
        max[i] = (uint64_t *)calloc(dt->num_levels, sizeof (uint64_t));
        count[i] = (uint64_t *)calloc(dt->num_levels, sizeof (uint64_t));
        avg[i] = (uint64_t *)calloc(dt->num_levels, sizeof (uint64_t));
        ranks[i] = (uint64_t *)calloc(dt->num_levels, sizeof (uint64_t));
    }

    if (dt->g->rank == 0)
        printf("\n[level] Dtree profile: #calls, #ranks, "
               "avg [min, max] (usecs)\n");
    for (int i = 0;  i < NTIMES;  i++) {
        for (int l = dt->num_levels - 1;  l >= 0;  l--) {
            r = a = 0;
            if (dt->my_level == l) {
                MPI_Reduce(&coll_times[i].min, &min[i][l], 1, MPI_LONG,
                           MPI_MIN, 0, MPI_COMM_WORLD);
                MPI_Reduce(&coll_times[i].max, &max[i][l], 1, MPI_LONG,
                           MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&coll_times[i].count, &count[i][l], 1, MPI_LONG,
                           MPI_SUM, 0, MPI_COMM_WORLD);
                if (coll_times[i].count > 0) {
                    a = coll_times[i].total / coll_times[i].count;
                    r = 1;
                }
            }
            else {
                u = UINT64_MAX;
                MPI_Reduce(&u, &min[i][l], 1, MPI_LONG,
                           MPI_MIN, 0, MPI_COMM_WORLD);
                u = 0;
                MPI_Reduce(&u, &max[i][l], 1, MPI_LONG,
                           MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&u, &count[i][l], 1, MPI_LONG,
                           MPI_SUM, 0, MPI_COMM_WORLD);
            }
            MPI_Reduce(&a, &avg[i][l], 1, MPI_LONG,
                       MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&r, &ranks[i][l], 1, MPI_LONG,
                       MPI_SUM, 0, MPI_COMM_WORLD);

            if (dt->g->rank == 0) {
                if (min[i][l] == UINT64_MAX)
                    min[i][l] = 0;
                if (ranks[i][l] > 0) {
                    avg[i][l] /= ranks[i][l];
                    count[i][l] /= ranks[i][l];
                }
                else
                    avg[i][l] = count[i][l] = 0;

                printf("[level %d] %s: %llu, %llu, %g [%g, %g]\n", l,
                       times_names[i], count[i][l], ranks[i][l],
                       avg[i][l] / cpu_mhz,
                       min[i][l] / cpu_mhz, max[i][l] / cpu_mhz);
            }
        }
    }

    for (int i = 0;  i < NTIMES;  i++) {
        free(min[i]);
        free(max[i]);
        free(count[i]);
        free(avg[i]);
        free(ranks[i]);
    }
#endif // PROFILE_DTREE

    _mm_free(dt->distrib_fractions);
    if (dt->num_children > 0) {
        _mm_free(dt->children_reqs);
        _mm_free(dt->children_req_bufs);
        _mm_free(dt->children);
    }
    _mm_free(dt);
}


/*  dtree_initwork() -- initialization: set up distribution fractions and
        initial work allocations (rank 0 has all the work, other ranks ask
        their parents, requests flow up, work flows down).
 */
int64_t dtree_initwork(dtree_t *dt, int64_t *first_item, int64_t *last_item)
{
    int i;

    /* wait for all ranks to get here */
    MPI_Barrier(MPI_COMM_WORLD);

    /* set up child distribution fractions */
    init_distrib_fractions(dt);

    /* parents wait for children to ask for work and aggregate the requests */
    int16_t req_items = dt->min_distrib;
    int64_t work[2];
    if (dt->num_children > 0) {
        for (i = 0;  i < dt->num_children;  i++)
            MPI_Irecv(&dt->children_req_bufs[i], 1, MPI_SHORT, dt->children[i], 0,
                      MPI_COMM_WORLD, &dt->children_reqs[i]);
        MPI_Waitall(dt->num_children, dt->children_reqs, MPI_STATUSES_IGNORE);
        for (i = 0;  i < dt->num_children;  i++)
            req_items += dt->children_req_bufs[i];
    }

    /* all ranks except the root ask their parent for work */
    if (dt->g->rank != 0) {
        DTREE_TRACE(dt, "[%04d] initwork: asking [%04d] for %d work items\n",
                dt->g->rank, dt->parent, req_items);
        MPI_Irecv(&work, 2, MPI_LONG, dt->parent, 0, MPI_COMM_WORLD, &dt->parent_req);
        MPI_Send(&req_items, 1, MPI_SHORT, dt->parent, 0, MPI_COMM_WORLD);
        MPI_Wait(&dt->parent_req, MPI_STATUS_IGNORE);

        dt->next_work_item = dt->first_work_item = work[0];
        dt->last_work_item = work[1];

        DTREE_TRACE(dt, "[%04d] init: got %llu items (%llu to %llu)\n",
                dt->g->rank, dt->last_work_item - dt->first_work_item,
                dt->first_work_item, dt->last_work_item);
    }

    /* determine how much initial work is available */
    int64_t avail_items = dt->last_work_item - dt->first_work_item;

    /* parents scale the initial distribution, if possible */
    if (dt->num_children > 0) {
        if (avail_items * dt->first >= req_items)
            avail_items *= dt->first;
        else if (avail_items >= req_items)
            DTREE_TRACE(dt, "[%04d] init: not enough work, discarding `first` scaling\n",
                        dt->g->rank);
        else
            DTREE_TRACE(dt, "[%04d] init: not enough work, some children will idle!\n",
                        dt->g->rank);
    }

    /* parents distribute work to their children */
    int64_t this_child;
    for (i = 0;  i < dt->num_children;  i++) {
        this_child = MIN(dt->last_work_item - dt->next_work_item,
                         MAX(avail_items * dt->distrib_fractions[i+1], dt->min_distrib));

        if (this_child > 0) {
            work[0] = dt->next_work_item;
            dt->next_work_item += this_child;
            work[1] = dt->next_work_item;
        }
        else
            work[0] = work[1] = 0;

        DTREE_TRACE(dt, "[%04d] init: feeding %ld items to [%04d]\n",
                dt->g->rank, this_child, dt->children[i]);

        MPI_Send(work, 2, MPI_LONG, dt->children[i], 0, MPI_COMM_WORLD);
        if (this_child > 0)
            MPI_Irecv(&dt->children_req_bufs[i], 1, MPI_SHORT, dt->children[i], 0,
                      MPI_COMM_WORLD, &dt->children_reqs[i]);
    }

    /* all working ranks (parents and children) keep themselves some work */
    int64_t my_items = 0;
    if (dt->num_children == 0  ||  dt->parents_work) {
        my_items = MIN(dt->last_work_item - dt->next_work_item, 
                       MAX(avail_items * dt->distrib_fractions[0], dt->min_distrib));
        if (my_items > 0) {
            *first_item = dt->next_work_item;
            dt->next_work_item += my_items;
            *last_item = dt->next_work_item;
        }
        else
            *first_item = *last_item = 0;
    }

    return my_items;
}


/*  dtree_getwork_aux() -- helper to get a block of work. Returns a
        number of work items to process, starting at *first_item. Zero
        returned means that the entire system is out of work.
 */
static int64_t dtree_getwork_aux(dtree_t *dt, int64_t *first_item, int64_t *last_item,
                       int child, int16_t req_items)
{
    int64_t avail_items, dist_items, num_items;

    PROFILE_START(dt, TIME_GETWORK);
    lock_acquire(dt->work_lock);

    /* are we completely out already? */
    if (dt->last_work_item == 0)
        goto outofwork;

    /* if we need more, ask */
    if (dt->next_work_item == dt->last_work_item) {
        if (dt->g->rank == 0)
            dt->first_work_item = dt->next_work_item = dt->last_work_item = 0;
        else {
            DTREE_TRACE(dt, "[%04d] asking [%04d] for work (had %llu to %llu, at %llu)\n",
                    dt->g->rank, dt->parent, dt->first_work_item, dt->last_work_item,
                    dt->next_work_item);

            PROFILE_START(dt, TIME_MPIWAIT);

            int64_t work[2];
            MPI_Irecv(&work, 2, MPI_LONG, dt->parent, 0, MPI_COMM_WORLD,
                      &dt->parent_req);
            req_items += dt->min_distrib;
            MPI_Send(&req_items, 1, MPI_SHORT, dt->parent, 0, MPI_COMM_WORLD);
            MPI_Wait(&dt->parent_req, MPI_STATUS_IGNORE);

            PROFILE_STAMP(dt, TIME_MPIWAIT);

            dt->next_work_item = dt->first_work_item = work[0];
            dt->last_work_item = work[1];

            DTREE_TRACE(dt, "[%04d] got %llu items (%llu to %llu)\n",
                    dt->g->rank, dt->last_work_item - dt->first_work_item,
                    dt->first_work_item, dt->last_work_item);
        }

        if (dt->last_work_item == 0) {
            DTREE_TRACE(dt, "[%04d] out of work\n", dt->g->rank);
            goto outofwork;
        }
    }

    /* Compute how many work items to send; this is KEY! dist_items is
       computed for the requesting child based on the available items
       and the child's sub-tree size.
     */
    avail_items = dt->last_work_item - dt->next_work_item;
    dist_items = avail_items * dt->rest * dt->distrib_fractions[child + 1];

    /* The minimum distribution is carried up the chain of requesting
       children in req_items. For the local rank, we have the minimum
       distribution.
     */
    if (child == -1) {
        if (dist_items < dt->min_distrib)
            dist_items = dt->min_distrib;
    }
    else {
        if (dist_items < req_items)
            dist_items = req_items;
    }
    num_items = MIN(avail_items, dist_items);

    *first_item = dt->next_work_item;
    dt->next_work_item += num_items;
    *last_item = dt->next_work_item;

    lock_release(dt->work_lock);
    PROFILE_STAMP(dt, TIME_GETWORK);

    return num_items;

outofwork:
    lock_release(dt->work_lock);
    PROFILE_STAMP(dt, TIME_GETWORK);

    *first_item = *last_item = 0;
    return 0;
}


/*  dtree_getwork() -- interface to get a block of work. Returns a
        number of work items to process, starting at *first_item. Zero
        returned means that the entire system is out of work.
 */
int64_t dtree_getwork(dtree_t *dt, int64_t *first_item, int64_t *last_item)
{
    /* if this is a parent and parents don't work, return no work */
    if (dt->num_children > 0  &&  !dt->parents_work) {
        *first_item = 0;
        *last_item = 0;
        return 0;
    }

    return dtree_getwork_aux(dt, first_item, last_item, -1, 0);
}


/*  dtree_run() -- feed children with work. Sending zeroes to a child
        tells it we're completely out of work. Must be called again if 1
        is returned.
 */
int dtree_run(dtree_t *dt)
{
    if (dt->num_children == 0)
        return 0;

    int index, flag;
    int16_t req_items;
    int64_t num_items, work[2];
    MPI_Request req;

    PROFILE_START(dt, TIME_RUN);
    PROFILE_START(dt, TIME_MPISEND);

    for (; ;) {
        MPI_Testany(dt->num_children, dt->children_reqs, &index, &flag,
                    MPI_STATUS_IGNORE);
        if (index == MPI_UNDEFINED) {
            PROFILE_STAMP(dt, TIME_RUN);
            return !flag;
        }

        req_items = dt->children_req_bufs[index];

        /* feed this child */
        num_items = dtree_getwork_aux(dt, &work[0], &work[1], index, req_items);

        DTREE_TRACE(dt, "[%04d] feeding %ld items to [%04d] (%ld to %ld)\n",
                dt->g->rank, num_items, dt->children[index], work[0], work[1]);

        MPI_Isend(work, 2, MPI_LONG, dt->children[index], 0, MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);

        PROFILE_STAMP(dt, TIME_MPISEND);

        /* if we sent some work, repost the irecv */
        if (num_items > 0)
            MPI_Irecv(&dt->children_req_bufs[index], 1, MPI_SHORT, dt->children[index],
                      0, MPI_COMM_WORLD, &dt->children_reqs[index]);
    }
}

