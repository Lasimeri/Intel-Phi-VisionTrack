/* live_depth.c: see live_depth.h. */
#include "live_depth.h"

#include <stdio.h>
#include <string.h>

#include "depth.h"
#include "img.h"
#include "session.h"
#include "still.h"
#include "vt_util.h"

#define HEAD_BYTES VT_DEPTH_Z_AT

struct live_depth {
    const struct vt_config *cfg;
    struct team *tm;
    int n;
    struct arena *a;
    struct vt_dma *dma;
    uint64_t host_buffers; /* card address of the first */
    struct still *st[VT_MAX_CAMERAS];
    struct depth *d;
    struct vt_plane nd, sd, nr, nl; /* normalized planes, and D's spread */
    char *scratch;
    size_t scratch_each;
    struct depth_view views[DEPTH_MAX_VIEWS];
    struct vt_depth_out *head;
    uint64_t geometry;
    int have_geometry;
    uint64_t seq;
    float f, cx, cy;
    uint64_t cycles_vectors[VT_MAX_CAMERAS], cycles_mean[VT_MAX_CAMERAS];
};

static void *arena_alloc(void *ctx, size_t bytes) { return arena_take(ctx, bytes, 64); }

static struct depth_cfg map_config(const struct vt_config *c, int threads)
{
    return (struct depth_cfg){
        .w = (int)c->width3 >> VT_DEPTH_LEVEL,
        .h = (int)c->height3 >> VT_DEPTH_LEVEL,
        .f = 1.0,
        .planes = (int)c->depth_planes,
        .z_near = (double)c->depth_near,
        .z_far = (double)c->depth_far,
        .agg_radius = (int)c->depth_agg,
        .trunc = c->depth_trunc,
        .tex_min = c->depth_tex,
        .conf_min = c->depth_conf,
        .threads = threads,
    };
}

static struct still_cfg still_config(const struct vt_config *c, int cam, int threads)
{
    const uint32_t w = cam == 0 ? c->width : cam == 1 ? c->width2 : c->width3, h = cam == 0 ? c->height : cam == 1 ? c->height2 : c->height3;
    return (struct still_cfg){.w = (int)w >> VT_DEPTH_LEVEL, .h = (int)h >> VT_DEPTH_LEVEL, .avg = (int)c->depth_avg, .motion_px = c->motion_px, .motion_grey = c->motion_grey, .threads = threads};
}

size_t ld_bytes(const struct vt_config *c, int threads)
{
    const struct depth_cfg m = map_config(c, threads);
    size_t bytes = depth_bytes(&m) + sizeof(struct live_depth) + HEAD_BYTES + (1u << 20);
    int widest = 0;
    for (int k = 0; k < VT_MAX_CAMERAS; k++) {
        const struct still_cfg s = still_config(c, k, threads);
        bytes += still_bytes(&s) + plane_bytes(s.w, s.h);
        widest = s.w > widest ? s.w : widest;
    }
    return bytes + plane_bytes(m.w, m.h) + (size_t)threads * depth_norm_scratch_bytes(widest);
}

struct live_depth *ld_create(const struct vt_config *c, struct team *tm, int threads, struct arena *a, struct vt_dma *dma)
{
    struct live_depth *l = arena_take(a, (sizeof *l + 63) & ~(size_t)63, 64);
    if (!l)
        return NULL;
    memset(l, 0, sizeof *l);
    l->cfg = c;
    l->tm = tm;
    l->n = threads;
    l->a = a;
    l->dma = dma;
    l->host_buffers = dma->region_addr + VT_PART_OFFSET + VT_DEPTH_OFFSET;
    const struct depth_cfg m = map_config(c, threads);
    l->d = depth_create(&m, arena_alloc, a);
    if (!l->d)
        return NULL;
    size_t at = (size_t)VT_DEPTH_Z_AT + 2u * (size_t)depth_stride(l->d) * (size_t)m.h * sizeof(float);
    int widest = 0;
    for (int k = 0; k < VT_MAX_CAMERAS; k++) {
        const struct still_cfg s = still_config(c, k, threads);
        l->st[k] = still_create(&s, arena_alloc, a);
        if (!l->st[k])
            return NULL;
        at += still_cell_bytes(l->st[k]);
        widest = s.w > widest ? s.w : widest;
    }
    if (at > VT_DEPTH_BYTES) {
        fprintf(stderr, "vtrackd: a depth map of %zu bytes does not fit its buffer\n", at);
        return NULL;
    }
    const struct still_cfg sr = still_config(c, 1, threads), sl = still_config(c, 0, threads);
    void *pd = arena_take(a, plane_bytes(m.w, m.h), 64), *ps = arena_take(a, plane_bytes(m.w, m.h), 64), *pr = arena_take(a, plane_bytes(sr.w, sr.h), 64), *pl = arena_take(a, plane_bytes(sl.w, sl.h), 64);
    l->scratch_each = depth_norm_scratch_bytes(widest);
    l->scratch = arena_take(a, (size_t)threads * l->scratch_each, 64);
    /* A block of this size at this alignment cannot cross a huge page: one DMA line, whose status comes after everything. */
    l->head = arena_take(a, HEAD_BYTES, HEAD_BYTES);
    if (!pd || !ps || !pr || !pl || !l->scratch || !l->head)
        return NULL;
    l->nd = plane_at(pd, m.w, m.h);
    l->sd = plane_at(ps, m.w, m.h);
    l->nr = plane_at(pr, sr.w, sr.h);
    l->nl = plane_at(pl, sl.w, sl.h);
    return l;
}

void ld_set_calib(struct live_depth *l, const struct vt_calib *c, uint64_t seq)
{
    l->have_geometry = c->valid && c->valid_d;
    l->geometry = seq;
    if (!l->have_geometry)
        return;
    const long double scale = (long double)(1 << VT_DEPTH_LEVEL);
    l->f = (float)((long double)c->fd / scale);
    l->cx = (float)((long double)c->cxd / scale);
    l->cy = (float)((long double)c->cyd / scale);
    depth_set_ref(l->d, (double)((long double)c->fd / scale), (double)((long double)c->cxd / scale), (double)((long double)c->cyd / scale));
    depth_views_from_calib(c, 1 << VT_DEPTH_LEVEL, &l->nr, &l->nl, l->views);
}

struct frame_job {
    struct still *s;
    const struct vt_plane *cur;
};

static void team_vectors(void *arg, int tid, int n)
{
    struct frame_job *j = arg;
    still_vectors(j->s, j->cur, tid, n);
}

static void team_update(void *arg, int tid, int n)
{
    struct frame_job *j = arg;
    still_update(j->s, j->cur, tid, n);
}

void ld_frame(struct live_depth *l, int cam, const struct vt_plane *cur)
{
    struct frame_job j = {l->st[cam], cur};
    const uint64_t t0 = vt_tsc();
    if (!still_first(j.s)) {
        team_run(l->tm, team_vectors, &j);
        img_fence();
    }
    const uint64_t t1 = vt_tsc();
    still_decide(j.s);
    team_run(l->tm, team_update, &j);
    img_fence();
    still_finish(j.s);
    const uint64_t t2 = vt_tsc();
    l->cycles_vectors[cam] = t1 - t0;
    l->cycles_mean[cam] = t2 - t1;
}

struct norm_job {
    const struct vt_plane *dst, *spread, *src;
    float eps;
    char *scratch;
    size_t each;
};

static void team_norm(void *arg, int tid, int n)
{
    const struct norm_job *j = arg;
    depth_normalize_part(j->dst, j->spread, j->src, j->eps, tid, n, j->scratch, j->each);
}

static void team_sweep(void *arg, int tid, int n) { depth_sweep(arg, tid, n); }
static void team_agg(void *arg, int tid, int n) { depth_aggregate(arg, tid, n); }
static void team_decide(void *arg, int tid, int n) { depth_decide(arg, tid, n); }

uint64_t ld_map(struct live_depth *l, const struct vt_msg *m, uint64_t status_addr)
{
    for (int k = 0; k < VT_MAX_CAMERAS; k++)
        if (still_first(l->st[k]))
            return 0;
    if (!l->have_geometry)
        return 0;
    const float eps = l->cfg->depth_eps;
    const struct norm_job a = {&l->nd, &l->sd, still_mean(l->st[2]), eps, l->scratch, l->scratch_each}, b = {&l->nr, NULL, still_mean(l->st[1]), eps, l->scratch, l->scratch_each}, c = {&l->nl, NULL, still_mean(l->st[0]), eps, l->scratch, l->scratch_each};
    const uint64_t t0 = vt_tsc();
    team_run(l->tm, team_norm, (void *)&a);
    team_run(l->tm, team_norm, (void *)&b);
    team_run(l->tm, team_norm, (void *)&c);
    img_fence();
    depth_set_views(l->d, &l->nd, &l->sd, l->views, 2);
    const uint64_t t1 = vt_tsc();
    team_run(l->tm, team_sweep, l->d);
    img_fence();
    const uint64_t t2 = vt_tsc();
    team_run(l->tm, team_agg, l->d);
    img_fence();
    const uint64_t t3 = vt_tsc();
    team_run(l->tm, team_decide, l->d);
    img_fence();
    const uint64_t t4 = vt_tsc();

    const struct depth_cfg map = map_config(l->cfg, l->n);
    const int stride = depth_stride(l->d);
    l->seq++;
    struct vt_depth_out *h = l->head;
    memset(h, 0, sizeof *h);
    h->seq = l->seq;
    h->instance = m->instance;
    h->acq_ns = m->acq_ns;
    h->geometry = l->geometry;
    h->w = (uint32_t)map.w;
    h->h = (uint32_t)map.h;
    h->stride = (uint32_t)stride;
    h->planes = (uint32_t)map.planes;
    h->z_near = l->cfg->depth_near;
    h->z_far = l->cfg->depth_far;
    h->f = l->f;
    h->cx = l->cx;
    h->cy = l->cy;
    h->views = 2;
    depth_counts(l->d, l->n, &h->textured, &h->confident, &h->known);
    h->cycles_normalize = t1 - t0;
    h->cycles_sweep = t2 - t1;
    h->cycles_aggregate = t3 - t2;
    h->cycles_decide = t4 - t3;

    /* Into the buffer the host is not reading: everything, then the announcement, in order on one channel. */
    const uint64_t host = l->host_buffers + ((l->seq - 1) & 1) * VT_DEPTH_BYTES;
    const uint64_t plane_b = (uint64_t)stride * (uint64_t)map.h * sizeof(float);
    uint64_t at = VT_DEPTH_Z_AT;
    if (!session_copy_out(l->dma, l->a, (void *)depth_z(l->d), host + at, plane_b))
        return UINT64_MAX;
    at += plane_b;
    if (!session_copy_out(l->dma, l->a, (void *)depth_conf(l->d), host + at, plane_b))
        return UINT64_MAX;
    at += plane_b;
    for (int k = 0; k < VT_MAX_CAMERAS; k++) {
        struct vt_motion *mo = &h->cam[k];
        still_motion(l->st[k], mo);
        mo->cell_at = (uint32_t)at;
        mo->cycles_vectors = l->cycles_vectors[k];
        mo->cycles_mean = l->cycles_mean[k];
        const size_t bytes = still_cell_bytes(l->st[k]);
        if (!session_copy_out(l->dma, l->a, (void *)still_cells(l->st[k]), host + at, bytes))
            return UINT64_MAX;
        at += bytes;
    }
    const uint64_t last = vt_dma_copy(l->dma, arena_phys(l->a, h), host, HEAD_BYTES, status_addr, l->seq);
    return last ? last : UINT64_MAX;
}
