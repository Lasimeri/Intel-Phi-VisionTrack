/* stars.c: see stars.h. */
#include "stars.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_AREA 80     /* pixels: more is a lit surface */
#define RADIUS 8        /* the ring a candidate's surroundings are sampled on */
#define NOISE_FLOOR 64.0 /* a quarter of a grey level: an integrated dark frame can show no step at all */
#define RECURSIVE_RADIUS 6
#define FIND_BLOB 26     /* half a window: holds the largest blurred disc seen */
#define FIND_MIN_GREY 4.0 /* grey levels above the surroundings */
#define FIND_MAX_AREA 2200 /* a disc 52 across; more is a lit surface */
#define CAT_MIN_FLUX 0.05 /* of the brightest: a faint companion only adds a wrong answer */
#define VOTE_RADIUS 2.5  /* pixels: lights agreeing on a shift */
#define RAIM_PX 1.5      /* a light further than this from the others' rotation is out */

static int cmp_u16(const void *a, const void *b)
{
    return (int)*(const uint16_t *)a - (int)*(const uint16_t *)b;
}

static int cmp_flux(const void *a, const void *b)
{
    const double x = ((const struct star *)a)->flux, y = ((const struct star *)b)->flux;
    return x > y ? -1 : x < y;
}

static int cmp_d(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* The median of a square ring just outside a box: the background under a light. */
static double ring_background(const uint16_t *g, int w, int h, int x0, int y0, int x1, int y1)
{
    uint16_t v[512];
    int n = 0;
    const int m = 3;
    for (int x = x0 - m; x <= x1 + m && n < 508; x += 1 + (x1 - x0 + 2 * m) / 64) {
        if (x < 0 || x >= w)
            continue;
        if (y0 - m >= 0)
            v[n++] = g[(size_t)(y0 - m) * (size_t)w + (size_t)x];
        if (y1 + m < h)
            v[n++] = g[(size_t)(y1 + m) * (size_t)w + (size_t)x];
    }
    for (int y = y0 - m; y <= y1 + m && n < 508; y += 1 + (y1 - y0 + 2 * m) / 64) {
        if (y < 0 || y >= h)
            continue;
        if (x0 - m >= 0)
            v[n++] = g[(size_t)y * (size_t)w + (size_t)(x0 - m)];
        if (x1 + m < w)
            v[n++] = g[(size_t)y * (size_t)w + (size_t)(x1 + m)];
    }
    if (!n)
        return 0.0;
    qsort(v, (size_t)n, sizeof v[0], cmp_u16);
    const uint16_t middle = v[n / 2];
    return (double)middle;
}

int star_detect(const uint16_t *g, int w, int h, double k, struct star *out, int max)
{
    const size_t n = (size_t)w * (size_t)h;
    uint16_t *steps = malloc(n * sizeof(uint16_t));
    uint8_t *on = calloc(n, 1);
    int *stack = malloc(n * sizeof(int));
    if (!steps || !on || !stack) {
        free(steps);
        free(on);
        free(stack);
        return 0;
    }
    /* The frame's noise: one pixel against its neighbour, robustly. */
    size_t m = 0;
    for (int y = 0; y < h; y += 2)
        for (int x = 0; x + 1 < w; x += 2) {
            const size_t at = (size_t)y * (size_t)w + (size_t)x;
            steps[m++] = (uint16_t)abs((int)g[at] - (int)g[at + 1]);
        }
    qsort(steps, m, sizeof(uint16_t), cmp_u16);
    const uint16_t middle_step = steps[m / 2];
    const double measured = (double)middle_step * 1.4826 / M_SQRT2;
    const double sigma = measured > NOISE_FLOOR ? measured : NOISE_FLOOR;
    free(steps);
    /* A candidate stands above the ring around it by k sigma, and that ring is DARK against it: all of it under half its brightness. A bright picture on a screen fails; a lamp in a dark room does not. */
    for (int y = RADIUS; y < h - RADIUS; y++)
        for (int x = RADIUS; x < w - RADIUS; x++) {
            const int c = g[(size_t)y * (size_t)w + (size_t)x];
            if ((double)c <= k * sigma)
                continue;
            uint16_t ring[16];
            for (int a = 0; a < 16; a++) {
                const double t = (double)a * M_PI / 8.0;
                ring[a] = g[(size_t)(y + (int)lrint(RADIUS * sin(t))) * (size_t)w + (size_t)(x + (int)lrint(RADIUS * cos(t)))];
            }
            qsort(ring, 16, sizeof(uint16_t), cmp_u16);
            if ((double)c - (double)ring[12] > k * sigma && 2 * (int)ring[15] < c)
                on[(size_t)y * (size_t)w + (size_t)x] = 1;
        }
    int nl = 0;
    for (size_t s = 0; s < n && nl < max; s++) {
        if (!on[s])
            continue;
        int sp = 0, area = 0;
        stack[sp++] = (int)s;
        on[s] = 0;
        int x0 = w, y0 = h, x1 = 0, y1 = 0;
        /* The component's pixels are kept on the stack's far end for the second pass. */
        static int pix[MAX_AREA + 1];
        while (sp) {
            const int p = stack[--sp];
            const int px = p % w, py = p / w;
            if (area <= MAX_AREA)
                pix[area < MAX_AREA ? area : MAX_AREA] = p;
            area++;
            x0 = px < x0 ? px : x0;
            x1 = px > x1 ? px : x1;
            y0 = py < y0 ? py : y0;
            y1 = py > y1 ? py : y1;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    const int qx = px + dx, qy = py + dy;
                    if (qx < 0 || qy < 0 || qx >= w || qy >= h)
                        continue;
                    const size_t q = (size_t)qy * (size_t)w + (size_t)qx;
                    if (on[q]) {
                        on[q] = 0;
                        stack[sp++] = (int)q;
                    }
                }
        }
        if (area > MAX_AREA)
            continue;
        const double bg = ring_background(g, w, h, x0, y0, x1, y1);
        double sx = 0, sy = 0, sw = 0;
        for (int i = 0; i < area; i++) {
            const double v = (double)g[pix[i]] - bg;
            const int col = pix[i] % w, row = pix[i] / w;
            if (v > 0.0) {
                sx += v * col;
                sy += v * row;
                sw += v;
            }
        }
        if (sw > 0.0) {
            out[nl].x = sx / sw;
            out[nl].y = sy / sw;
            out[nl].flux = sw;
            out[nl].area = area;
            nl++;
        }
    }
    free(on);
    free(stack);
    qsort(out, (size_t)nl, sizeof out[0], cmp_flux);
    return nl;
}

int star_find(const uint16_t *g, int w, int h, double x, double y, int radius, struct star *out)
{
    /* A lens focused on something near turns a far LED into a disc: measured on the camcorder, 40 pixels wide and 10 grey levels above its surroundings, where it was a 23-pixel point. So: the
     * peak of a 5 x 5 mean, not of one pixel; the background well outside a disc; the centroid over half of that peak: a disc's centre is as good as a point's. */
    const int cx = (int)lrint(x), cy = (int)lrint(y);
    const int margin = radius + FIND_BLOB + 8;
    if (cx - margin < 0 || cy - margin < 0 || cx + margin >= w || cy + margin >= h)
        return 0;
    const double bg = ring_background(g, w, h, cx - radius - FIND_BLOB, cy - radius - FIND_BLOB, cx + radius + FIND_BLOB, cy + radius + FIND_BLOB);
    double peak = 0.0;
    int px = cx, py = cy;
    for (int j = cy - radius; j <= cy + radius; j++)
        for (int i = cx - radius; i <= cx + radius; i++) {
            uint32_t s = 0;
            for (int b = -2; b <= 2; b++) {
                const uint16_t *row = g + (size_t)(j + b) * (size_t)w + (size_t)(i - 2);
                s += (uint32_t)row[0] + row[1] + row[2] + row[3] + row[4];
            }
            const double mean = (double)s / 25.0;
            if (mean > peak) {
                peak = mean;
                px = i;
                py = j;
            }
        }
    /* Standing clear of its surroundings, and those surroundings not a bright picture. */
    if (peak - bg < FIND_MIN_GREY * 257.0 || bg > 0.8 * peak)
        return 0;
    /* The half of the true peak near that place; then only what is connected to it: everything above half in a wide window swallowed neighbouring lights, measured: R's residual went from 0.40 to 2.54 px. */
    double top = peak;
    int tx = px, ty = py;
    for (int j = py - 2; j <= py + 2; j++)
        for (int i = px - 2; i <= px + 2; i++) {
            const double raw = (double)g[(size_t)j * (size_t)w + (size_t)i];
            if (raw > top) {
                top = raw;
                tx = i;
                ty = j;
            }
        }
    const double half = bg + 0.5 * (top - bg);
    enum { SIDE = 2 * FIND_BLOB + 1 };
    static uint8_t visited[SIDE * SIDE];
    static int queue[SIDE * SIDE];
    memset(visited, 0, sizeof visited);
    int head = 0, tail = 0;
    double sx = 0, sy = 0, sw = 0;
    int area = 0;
    queue[tail++] = (FIND_BLOB * SIDE) + FIND_BLOB;
    visited[queue[0]] = 1;
    while (head < tail) {
        const int at = queue[head++];
        const int rel_x = at % SIDE - FIND_BLOB, rel_y = at / SIDE - FIND_BLOB;
        const int i = tx + rel_x, j = ty + rel_y;
        const double raw = (double)g[(size_t)j * (size_t)w + (size_t)i];
        const double v = raw - bg;
        sx += v * i;
        sy += v * j;
        sw += v;
        area++;
        static const int nx[4] = {1, -1, 0, 0}, ny[4] = {0, 0, 1, -1};
        for (int d = 0; d < 4; d++) {
            const int qx = rel_x + nx[d], qy = rel_y + ny[d];
            if (qx < -FIND_BLOB || qx > FIND_BLOB || qy < -FIND_BLOB || qy > FIND_BLOB)
                continue;
            const int q = (qy + FIND_BLOB) * SIDE + qx + FIND_BLOB;
            if (visited[q])
                continue;
            visited[q] = 1;
            if ((double)g[(size_t)(ty + qy) * (size_t)w + (size_t)(tx + qx)] >= half)
                queue[tail++] = q;
        }
    }
    if (sw <= 0.0 || area > FIND_MAX_AREA)
        return 0;
    out->x = sx / sw;
    out->y = sy / sw;
    out->flux = sw;
    out->area = area;
    return 1;
}

/* Lost in space: every light in the frame against every catalogue light, a vote for the shift between them. The shift with the most votes is what the pattern says, whatever the pose was. */
static int vote_shift(const struct star_cat *c, const struct star *seen, int ns, double *dx, double *dy)
{
    int best = 0;
    double bx = 0, by = 0;
    for (int a = 0; a < c->n; a++)
        for (int s = 0; s < ns; s++) {
            const double sx = seen[s].x - c->s[a].x, sy = seen[s].y - c->s[a].y;
            int votes = 0;
            double ax = 0, ay = 0;
            for (int i = 0; i < c->n; i++)
                for (int j = 0; j < ns; j++) {
                    const double ex = seen[j].x - c->s[i].x - sx, ey = seen[j].y - c->s[i].y - sy;
                    if (ex * ex + ey * ey < VOTE_RADIUS * VOTE_RADIUS) {
                        votes++;
                        ax += sx + ex;
                        ay += sy + ey;
                        break;
                    }
                }
            if (votes > best) {
                best = votes;
                bx = ax / votes;
                by = ay / votes;
            }
        }
    *dx = bx;
    *dy = by;
    return best;
}

/* Solves A x = b, 3 x 3, by elimination with pivoting. 0 when singular. */
static int solve3(double A[3][3], double b[3], double *x)
{
    for (int i = 0; i < 3; i++) {
        int p = i;
        for (int r = i + 1; r < 3; r++)
            if (fabs(A[r][i]) > fabs(A[p][i]))
                p = r;
        if (fabs(A[p][i]) < 1e-14)
            return 0;
        for (int c = 0; c < 3; c++) {
            const double t = A[i][c];
            A[i][c] = A[p][c];
            A[p][c] = t;
        }
        const double t = b[i];
        b[i] = b[p];
        b[p] = t;
        for (int r = i + 1; r < 3; r++) {
            const double m = A[r][i] / A[i][i];
            for (int c = i; c < 3; c++)
                A[r][c] -= m * A[i][c];
            b[r] -= m * b[i];
        }
    }
    for (int i = 2; i >= 0; i--) {
        double s = b[i];
        for (int c = i + 1; c < 3; c++)
            s -= A[i][c] * x[c];
        x[i] = s / A[i][i];
    }
    return 1;
}

/* Where a rotation (small) takes a bearing, in normalized image coordinates: the two rows of the image-motion field of a rotating camera. */
static void motion_rows(double u, double v, double f, double row_x[3], double row_y[3])
{
    /* d(x)/f = wy * (1 + u^2) - wx * u v - wz * v;  d(y)/f = wy * u v - wx * (1 + v^2) + wz * u */
    row_x[0] = -f * u * v;
    row_x[1] = f * (1.0 + u * u);
    row_x[2] = -f * v;
    row_y[0] = -f * (1.0 + v * v);
    row_y[1] = f * u * v;
    row_y[2] = f * u;
}

void star_rotation(double wx, double wy, double wz, double *R)
{
    const double th = sqrt(wx * wx + wy * wy + wz * wz);
    const double A = th < 1e-12 ? 1.0 : sin(th) / th, B = th < 1e-12 ? 0.5 : (1.0 - cos(th)) / (th * th);
    R[0] = 1.0 - B * (wy * wy + wz * wz);
    R[1] = -A * wz + B * wx * wy;
    R[2] = A * wy + B * wx * wz;
    R[3] = A * wz + B * wx * wy;
    R[4] = 1.0 - B * (wx * wx + wz * wz);
    R[5] = -A * wx + B * wy * wz;
    R[6] = -A * wy + B * wx * wz;
    R[7] = A * wx + B * wy * wz;
    R[8] = 1.0 - B * (wx * wx + wy * wy);
}

void star_correct_pair(const double *R, const double *T, const double *Oa, const double *Ob, double *Rout, double *Tout)
{
    /* R' = Ob R Oa' */
    double t[9];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            t[3 * r + c] = R[3 * r] * Oa[3 * c] + R[3 * r + 1] * Oa[3 * c + 1] + R[3 * r + 2] * Oa[3 * c + 2];
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            Rout[3 * r + c] = Ob[3 * r] * t[c] + Ob[3 * r + 1] * t[3 + c] + Ob[3 * r + 2] * t[6 + c];
    for (int r = 0; r < 3; r++)
        Tout[r] = Ob[3 * r] * T[0] + Ob[3 * r + 1] * T[1] + Ob[3 * r + 2] * T[2];
}

void star_track(const struct star_cat *c, const uint16_t *g, int w, int h, double f, double hint_dx, double hint_dy, int allow_lost_in_space, struct star_fix *fix)
{
    memset(fix, 0, sizeof *fix);
    fix->known = c->n;
    if (!c->n)
        return;
    const double cx = 0.5 * (w - 1), cy = 0.5 * (h - 1);
    static struct star found[STAR_MAX];
    int have[STAR_MAX];
    int seen = 0;
    double hx = hint_dx, hy = hint_dy;
    for (int pass = 0; pass < 2; pass++) {
        seen = 0;
        for (int i = 0; i < c->n; i++) {
            have[i] = star_find(g, w, h, c->s[i].x + hx, c->s[i].y + hy, RECURSIVE_RADIUS, &found[i]);
            seen += have[i];
        }
        if (seen * 2 >= c->n || pass || !allow_lost_in_space)
            break;
        /* Most of them are not where they should be: lost in space. Decided by the pattern over the whole frame. */
        static struct star all[STAR_MAX * 2];
        const int na = star_detect(g, w, h, 6.0, all, STAR_MAX * 2);
        double vx, vy;
        const int votes = vote_shift(c, all, na, &vx, &vy);
        /* One light against one light says nothing about which is which: a pattern needs two, unless the catalogue has only one. */
        if (votes >= (c->n >= 2 ? 2 : 1)) {
            hx = vx;
            hy = vy;
            fix->lost_in_space = 1;
        } else {
            break;
        }
    }
    fix->seen = seen;
    if (!seen)
        return;
    /* Median shift, then RAIM on it: a light far from the others' shift by more than a rotation could explain is out. */
    double xs[STAR_MAX], ys[STAR_MAX];
    int m = 0;
    for (int i = 0; i < c->n; i++)
        if (have[i]) {
            xs[m] = found[i].x - c->s[i].x;
            ys[m] = found[i].y - c->s[i].y;
            m++;
        }
    qsort(xs, (size_t)m, sizeof xs[0], cmp_d);
    qsort(ys, (size_t)m, sizeof ys[0], cmp_d);
    fix->dx = xs[m / 2];
    fix->dy = ys[m / 2];
    /* Attitude: least squares on the rotation, twice: the second time without the lights the first left far out. */
    int use[STAR_MAX];
    for (int i = 0; i < c->n; i++)
        use[i] = have[i];
    double wvec[3] = {0, 0, 0};
    /* Up to five rounds, one light out each round, the worst: RAIM's own way of isolating a fault. It never takes the set below two, and a set that still disagrees is reported as such by its residual. */
    for (int iter = 0; iter < 5; iter++) {
        double ATA[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, ATb[3] = {0, 0, 0};
        int n = 0;
        double su = 0, sv = 0, su2 = 0, sv2 = 0;
        for (int i = 0; i < c->n; i++) {
            if (!use[i])
                continue;
            const double u = (c->s[i].x - cx) / f, v = (c->s[i].y - cy) / f;
            double rx[3], ry[3];
            motion_rows(u, v, f, rx, ry);
            const double ex = found[i].x - c->s[i].x, ey = found[i].y - c->s[i].y;
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++)
                    ATA[a][b] += rx[a] * rx[b] + ry[a] * ry[b];
                ATb[a] += rx[a] * ex + ry[a] * ey;
            }
            su += u;
            sv += v;
            su2 += u * u;
            sv2 += v * v;
            n++;
        }
        fix->used = n;
        if (!n)
            return;
        /* Roll needs lights spread apart: without that it is held at zero, not guessed. */
        const double spread = (su2 - su * su / n) + (sv2 - sv * sv / n);
        fix->roll_known = n >= 2 && spread > 1e-5;
        if (!fix->roll_known) {
            const double huge = 1e6 * (f * f + ATA[0][0] + ATA[1][1]);
            ATA[2][2] += huge;
        }
        double A[3][3], bb[3];
        memcpy(A, ATA, sizeof A);
        memcpy(bb, ATb, sizeof bb);
        if (!solve3(A, bb, wvec))
            return;
        /* GDOP: the rotation's uncertainty for unit centroid noise: the diagonal of the inverse normal matrix. */
        double trace = 0.0;
        for (int d = 0; d < 3; d++) {
            double A2[3][3], e[3] = {0, 0, 0}, x[3];
            memcpy(A2, ATA, sizeof A2);
            e[d] = 1.0;
            if (solve3(A2, e, x))
                trace += x[d];
        }
        fix->gdop = sqrt(trace) * 180.0 / M_PI;
        /* Residuals. */
        double sum = 0.0, worst = 0.0;
        int worst_i = -1;
        for (int i = 0; i < c->n; i++) {
            if (!use[i])
                continue;
            const double u = (c->s[i].x - cx) / f, v = (c->s[i].y - cy) / f;
            double rx[3], ry[3];
            motion_rows(u, v, f, rx, ry);
            const double ex = found[i].x - c->s[i].x - (rx[0] * wvec[0] + rx[1] * wvec[1] + rx[2] * wvec[2]);
            const double ey = found[i].y - c->s[i].y - (ry[0] * wvec[0] + ry[1] * wvec[1] + ry[2] * wvec[2]);
            const double e2 = ex * ex + ey * ey;
            sum += e2;
            if (e2 > worst) {
                worst = e2;
                worst_i = i;
            }
        }
        fix->rms_px = sqrt(sum / n);
        if (sqrt(worst) <= RAIM_PX || n <= 2 || worst_i < 0)
            break;
        use[worst_i] = 0;
    }
    fix->wx = wvec[0];
    fix->wy = wvec[1];
    fix->wz = wvec[2];
}

int star_catalogue(const struct star *a, int na, const struct star *b, int nb, double max_shift, int min_area, double min_separation, struct star_cat *out)
{
    out->n = 0;
    /* a is brightest first. Measured: a 4 % companion made a false "camera turned, roll -1.174 deg". */
    const double floor_flux = na ? a[0].flux * CAT_MIN_FLUX : 0.0;
    for (int i = 0; i < na && out->n < STAR_MAX; i++) {
        if (a[i].area < min_area || a[i].flux < floor_flux)
            continue;
        /* A close double, or a member of a dense field, cannot be told from its neighbour once anything moves: a star catalogue leaves those out. Measured: a keyboard's backlight made every light find the wrong key at a 0.5 degree roll. */
        int crowded = 0;
        for (int j = 0; j < na; j++) {
            const double dx = a[i].x - a[j].x, dy = a[i].y - a[j].y;
            /* A much fainter neighbour cannot be mistaken for it: the rule threw out the room's real LED for its dim companion. */
            if (j != i && dx * dx + dy * dy < min_separation * min_separation && a[j].flux * 3.0 > a[i].flux)
                crowded = 1;
        }
        if (crowded)
            continue;
        for (int j = 0; j < nb; j++) {
            const double dx = a[i].x - b[j].x, dy = a[i].y - b[j].y;
            if (dx * dx + dy * dy <= max_shift * max_shift && b[j].area >= min_area) {
                struct star *s = &out->s[out->n++];
                s->x = 0.5 * (a[i].x + b[j].x);
                s->y = 0.5 * (a[i].y + b[j].y);
                s->flux = 0.5 * (a[i].flux + b[j].flux);
                s->area = a[i].area;
                break;
            }
        }
    }
    return out->n;
}

int star_cat_write(const char *path, const struct star_cat *c)
{
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    fprintf(f, "# x y flux area: a camera's fixed lights, in its own image, from repeated long exposures\n");
    for (int i = 0; i < c->n; i++)
        fprintf(f, "%.3f %.3f %.0f %d\n", c->s[i].x, c->s[i].y, c->s[i].flux, c->s[i].area);
    fclose(f);
    return 0;
}

int star_cat_read(const char *path, struct star_cat *c)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    char line[128];
    c->n = 0;
    while (c->n < STAR_MAX && fgets(line, sizeof line, f)) {
        if (line[0] == '#')
            continue;
        struct star *s = &c->s[c->n];
        if (sscanf(line, "%lf %lf %lf %d", &s->x, &s->y, &s->flux, &s->area) == 4)
            c->n++;
    }
    fclose(f);
    return 0;
}

uint16_t *star_read_pgm(const char *path, int *w, int *h)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return NULL;
    }
    int maxv;
    if (fscanf(f, "P5 %d %d %d", w, h, &maxv) != 3 || (maxv != 255 && maxv != 65535)) {
        fprintf(stderr, "%s: not an 8-bit or 16-bit PGM\n", path);
        fclose(f);
        return NULL;
    }
    fgetc(f);
    const size_t n = (size_t)*w * (size_t)*h, bytes = maxv == 255 ? 1 : 2;
    uint8_t *raw = malloc(n * bytes);
    uint16_t *g = malloc(n * sizeof(uint16_t));
    if (!raw || !g || fread(raw, bytes, n, f) != n) {
        fprintf(stderr, "%s: short\n", path);
        fclose(f);
        free(raw);
        free(g);
        return NULL;
    }
    fclose(f);
    for (size_t i = 0; i < n; i++)
        g[i] = maxv == 255 ? (uint16_t)(raw[i] * 257u) : (uint16_t)((raw[2 * i] << 8) | raw[2 * i + 1]);
    free(raw);
    return g;
}
