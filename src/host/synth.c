/* synth.c: see synth.h. */
#include "synth.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

static const char *const names[SC_COUNT] = {"translate", "shapes", "camera", "brightness", "noise", "occlusion", "loss", "leave", "helix", "linear3d", "repeat", "flat", "glitch", "bigbox", "hold", "smallbox", "stillstart", "rotate", "swing", "comeback", "offbox", "jointed", "offjointed", "passby", "dart", "dual", "dualbox"};

const char *scene_name(enum scene_id id) { return names[id]; }

enum scene_id scene_by_name(const char *name)
{
    for (int i = 0; i < SC_COUNT; i++)
        if (!strcmp(name, names[i]))
            return (enum scene_id)i;
    return SC_COUNT;
}

void scene_setup(struct scene *s, enum scene_id id, int w, int h, int nframes, double fps)
{
    s->id = id;
    s->name = names[id];
    s->w = w;
    s->h = h;
    s->nframes = nframes;
    s->fps = fps;
    /* A focal length that gives every size the same field of view: 53 degrees across. */
    s->f = (double)w;
    s->cx = 0.5 * (double)(w - 1);
    s->cy = 0.5 * (double)(h - 1);
    s->baseline = 0.12;
    s->stereo = 1;
    s->select_frame = 5;
    s->samples = w > 1400 ? 2 : 3;
    s->shutter = id == SC_JOINTED || id == SC_OFFJOINTED || id == SC_DART ? 0.5 : 0.0;
    if (id == SC_DUAL || id == SC_DUALBOX) {
        /* Camera 1 stands at C = (1.2, 0.1, 0.2) m, turned by 17.5 degrees about y toward the scene, with 70 % of the focal length. */
        const double th = 17.5 * M_PI / 180.0, ct = cos(th), st = sin(th);
        const double R[9] = {ct, 0.0, st, 0.0, 1.0, 0.0, -st, 0.0, ct};
        const double C[3] = {1.2, 0.1, 0.2};
        memcpy(s->b_R, R, sizeof R);
        for (int i = 0; i < 3; i++)
            s->b_T[i] = -(R[3 * i] * C[0] + R[3 * i + 1] * C[1] + R[3 * i + 2] * C[2]);
        s->b_f = 0.7 * s->f;
        s->b_delay_s = 0.011;
    }
}

/* Scene geometry as functions of time, in seconds. Numbers are scale-free: distances in metres at a field of view that fixes the pixels. */
struct motion {
    double X, Y, Z, Vx, Vy, Vz;
    double halfx, halfy;
    int disc;
    double camX, camY;
    double occ_x, occ_halfx, occ_y, occ_halfy, occ_z; /* occluder at world (occ_x, occ_y), 0 half = none */
    double gain, offset;
    double noise;
    double second_x; /* a second, independent moving rectangle's X, in shapes */
    double spin;     /* the object's turn in the image plane, radians */
    int parts;       /* a thing of parts: each where its own motion puts it, relative to X, Y */
    double px[6], py[6], pspin[6];
};

/* Metres along the way at time t: a little still, on, a stop, on, and stopped. */
static inline double walk(double t, double v)
{
    const double a = t < 0.5 ? 0.0 : t < 1.7 ? t - 0.5 : 1.2, b = t < 2.7 ? 0.0 : t < 3.7 ? t - 2.7 : 1.0;
    return v * (a + b);
}

/* Six parts over a rectangle of half size (hx, hy), each wandering by up to 40 % of its own width and turning by up to 20 degrees, none in step with another. Across the parts the wandering averages to about nothing, but
 * at any moment no two of them agree: at 3 rad/s and 16 px a part moves up to 1.6 px a frame against its neighbour, as much as the walk itself. */
static void place_parts(struct motion *m, double t, double hx, double hy)
{
    static const double fu[6] = {2.1, 3.3, 2.7, 3.7, 2.3, 3.1}, fv[6] = {2.9, 2.2, 3.5, 2.6, 3.9, 2.4}, fs[6] = {1.7, 2.6, 2.1, 3.0, 1.9, 2.8}, ph[6] = {0.0, 1.1, 2.3, 3.7, 4.4, 5.6};
    m->parts = 6;
    m->halfx = hx;
    m->halfy = hy;
    for (int i = 0; i < 6; i++) {
        const double rx = (i % 3 - 1) * 0.66 * hx, ry = (i < 3 ? -0.5 : 0.5) * hy, wander = 0.4 * 0.34 * hx;
        m->px[i] = rx + wander * sin(fu[i] * t + ph[i]);
        m->py[i] = ry + wander * sin(fv[i] * t + 1.7 * ph[i]);
        m->pspin[i] = 0.35 * sin(fs[i] * t + 2.3 * ph[i]);
    }
}

/* A point of the world, relative to the object's centre, in the object's own axes. */
static inline void unturn(const struct motion *m, double *lx, double *ly)
{
    if (m->spin != 0.0) {
        const double c = cos(m->spin), s = sin(m->spin), x = *lx, y = *ly;
        *lx = c * x + s * y;
        *ly = c * y - s * x;
    }
}

/* The background as busy as the object, so that features on it are as good as features on the object. */
static inline int busy_background(const struct scene *s) { return s->id == SC_DUALBOX || s->id == SC_BIGBOX || s->id == SC_SMALLBOX || s->id == SC_STILLSTART || s->id == SC_ROTATE || s->id == SC_SWING || s->id == SC_COMEBACK || s->id == SC_OFFBOX || s->id == SC_JOINTED || s->id == SC_OFFJOINTED || s->id == SC_PASSBY || s->id == SC_DART; }

static void motion_at(const struct scene *s, double t, struct motion *m)
{
    memset(m, 0, sizeof *m);
    m->Z = 4.0;
    m->halfx = 0.5;
    m->halfy = 0.4;
    m->gain = 1.0;
    switch (s->id) {
    case SC_TRANSLATE:
        m->X = -0.9 + 0.30 * t;
        m->Y = -0.2 + 0.11 * t;
        m->Vx = 0.30;
        m->Vy = 0.11;
        break;
    case SC_SHAPES:
        m->X = 0.9 * sin(0.9 * t);
        m->Y = 0.4 * sin(1.7 * t);
        m->Vx = 0.81 * cos(0.9 * t);
        m->Vy = 0.68 * cos(1.7 * t);
        m->disc = 1;
        m->second_x = 0.9 * cos(0.5 * t);
        break;
    case SC_CAMERA:
        m->X = 0.2;
        m->camX = 0.8 * sin(0.8 * t);
        m->camY = 0.3 * sin(1.3 * t);
        m->Vx = -0.64 * cos(0.8 * t);
        m->Vy = -0.39 * cos(1.3 * t);
        break;
    case SC_BRIGHTNESS:
        m->X = -0.6 + 0.2 * t;
        m->Vx = 0.2;
        m->gain = 1.0 + 0.25 * sin(1.3 * t);
        m->offset = 16.0 * sin(0.7 * t);
        break;
    case SC_BIGBOX:
        m->X = -0.6 + 0.3 * t;
        m->Vx = 0.3;
        break;
    case SC_HOLD:
        m->noise = 4.0;
        break;
    case SC_SMALLBOX:
        m->X = -0.6 + 0.3 * t;
        m->Vx = 0.3;
        m->halfx = 0.25;
        m->halfy = 0.2;
        break;
    case SC_STILLSTART:
        m->X = t < 2.0 ? -0.3 : -0.3 + 0.3 * (t - 2.0);
        m->Vx = t < 2.0 ? 0.0 : 0.3;
        break;
    case SC_ROTATE:
        m->X = -0.6 + 0.2 * t;
        m->Vx = 0.2;
        m->spin = 30.0 * M_PI / 180.0 * t;
        break;
    case SC_SWING:
        m->X = 1.2 * sin(3.0 * t);
        m->Vx = 3.6 * cos(3.0 * t);
        break;
    case SC_COMEBACK:
        /* A little still; out to the right at 21 px a frame; away for 3 s; in from the left at 5 px a frame, on to the middle. */
        if (t < 0.5) {
            m->X = -0.6;
        } else if (t < 1.4) {
            m->X = -0.6 + 4.0 * (t - 0.5);
            m->Vx = 4.0;
        } else if (t < 4.4) {
            m->X = 8.0;
        } else {
            const double back = -3.2 + 1.0 * (t - 4.4);
            m->X = back < 0.2 ? back : 0.2;
            m->Vx = back < 0.2 ? 1.0 : 0.0;
        }
        break;
    case SC_OFFBOX:
        /* 40 x 32 px half size at 640 wide; the box 200 x 160, its centre 150 px to the left. */
        m->halfx = 0.25;
        m->halfy = 0.2;
        m->X = (370.0 - s->cx * 640.0 / s->w) / 160.0 + walk(t, 0.6);
        m->Y = 0.06 * sin(1.3 * t);
        m->Vx = (walk(t + 0.001, 0.6) - walk(t - 0.001, 0.6)) / 0.002;
        m->Vy = 0.078 * cos(1.3 * t);
        break;
    case SC_JOINTED:
        place_parts(m, t, 0.5, 0.4);
        m->X = (210.0 - s->cx * 640.0 / s->w) / 160.0 + walk(t, 0.6);
        m->Y = 0.06 * sin(1.3 * t);
        m->Vx = (walk(t + 0.001, 0.6) - walk(t - 0.001, 0.6)) / 0.002;
        m->Vy = 0.078 * cos(1.3 * t);
        m->noise = 2.0;
        break;
    case SC_OFFJOINTED:
        /* 48 x 40 px half size; the box 4.5 times that, its centre 160 px to the left: the thing's nearest edge 52 px outside the box's middle half. */
        place_parts(m, t, 0.3, 0.25);
        m->X = (376.0 - s->cx * 640.0 / s->w) / 160.0 + walk(t, 0.6);
        m->Y = 0.06 * sin(1.3 * t);
        m->Vx = (walk(t + 0.001, 0.6) - walk(t - 0.001, 0.6)) / 0.002;
        m->Vy = 0.078 * cos(1.3 * t);
        m->noise = 2.0;
        break;
    case SC_DART:
        m->halfx = 0.25;
        m->halfy = 0.2;
        m->X = 1.3 * sin(4.0 * t);
        m->Vx = 5.2 * cos(4.0 * t);
        m->noise = 2.0;
        break;
    case SC_PASSBY:
        /* The other thing is nearer, under the selected one and clear of it, inside the box: about 4.8 px a frame, in the box from about 1.2 s to 3.9 s. */
        m->noise = 2.0;
        m->occ_x = -1.4 + 0.56 * t;
        m->occ_y = 0.39;
        m->occ_halfx = 0.2;
        m->occ_halfy = 0.1;
        m->occ_z = 2.5;
        break;
    case SC_NOISE:
        m->X = -0.6 + 0.2 * t;
        m->Y = 0.15 * sin(1.1 * t);
        m->Vx = 0.2;
        m->Vy = 0.165 * cos(1.1 * t);
        m->noise = 4.0;
        break;
    case SC_OCCLUSION: {
        m->X = -0.4 + 0.08 * t;
        m->Vx = 0.08;
        /* A bar as wide as half the object sweeps over its upper half, nearer the camera. */
        m->occ_x = -1.6 + 0.55 * t;
        m->occ_halfx = 0.26;
        m->occ_y = -0.2;
        m->occ_halfy = 0.24;
        m->occ_z = 2.5;
        break;
    }
    case SC_LOSS: {
        m->X = -0.6 + 0.12 * t;
        m->Vx = 0.12;
        /* A block covers the object completely from 2.0 s to 3.0 s. */
        m->occ_halfy = 1.2;
        m->occ_z = 2.5;
        m->occ_y = 0.0;
        m->occ_halfx = 0.9;
        m->occ_x = (t >= 2.0 && t < 3.0) ? 0.0 + m->X * 2.5 / 4.0 : 100.0;
        break;
    }
    case SC_LEAVE:
        m->X = -0.5 + 0.55 * t;
        m->Vx = 0.55;
        break;
    case SC_HELIX:
        m->X = 0.8 * cos(1.2 * t);
        m->Y = 0.5 * sin(1.2 * t);
        m->Z = 4.0 + 0.25 * t;
        m->Vx = -0.96 * sin(1.2 * t);
        m->Vy = 0.6 * cos(1.2 * t);
        m->Vz = 0.25;
        break;
    case SC_LINEAR3D:
        m->X = -0.8 + 0.25 * t;
        m->Y = -0.3 + 0.1 * t;
        m->Z = 3.2 + 0.15 * t;
        m->Vx = 0.25;
        m->Vy = 0.1;
        m->Vz = 0.15;
        break;
    case SC_REPEAT:
    case SC_GLITCH:
        m->X = -0.3 + 0.1 * t;
        m->Vx = 0.1;
        break;
    case SC_DUALBOX: {
        const double tm = t < 2.0 ? 0.0 : t - 2.0, go = t < 2.0 ? 0.0 : 1.0;
        m->X = -0.5 + 0.2 * tm;
        m->Y = 0.15 * sin(1.1 * tm);
        m->Z = 3.6 + 0.2 * tm;
        m->Vx = 0.2 * go;
        m->Vy = 0.165 * cos(1.1 * tm) * go;
        m->Vz = 0.2 * go;
        break;
    }
    case SC_DUAL:
        m->X = -0.5 + 0.2 * t;
        m->Y = 0.15 * sin(1.1 * t);
        m->Z = 3.6 + 0.2 * t;
        m->Vx = 0.2;
        m->Vy = 0.165 * cos(1.1 * t);
        m->Vz = 0.2;
        break;
    case SC_FLAT:
        m->X = -0.3 + 0.1 * t;
        m->Vx = 0.1;
        break;
    default:
        break;
    }
}

/* ---- texture: value noise over world metres ---- */

static inline uint32_t hash2(int64_t x, int64_t y, uint32_t seed)
{
    uint64_t h = (uint64_t)x * 0x9E3779B97F4A7C15ull ^ (uint64_t)y * 0xC2B2AE3D27D4EB4Full ^ seed * 0x165667B19E3779F9ull;
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return (uint32_t)h;
}

static inline double vnoise(double x, double y, uint32_t seed)
{
    const double fx0 = floor(x), fy0 = floor(y);
    const int64_t xi = (int64_t)fx0, yi = (int64_t)fy0;
    const double fx = x - fx0, fy = y - fy0;
    /* Smoothstep: continuous slope, so sub-pixel motion moves the image smoothly. */
    const double sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
    const double a = (hash2(xi, yi, seed) & 255), b = (hash2(xi + 1, yi, seed) & 255), c = (hash2(xi, yi + 1, seed) & 255), d = (hash2(xi + 1, yi + 1, seed) & 255);
    return (a + (b - a) * sx) * (1 - sy) + (c + (d - c) * sx) * sy;
}

/* Grey 0 .. 255. wavelength in metres. */
static inline double texture(double x, double y, double wl, uint32_t seed)
{
    return 0.55 * vnoise(x / wl, y / wl, seed) + 0.30 * vnoise(x / (wl * 0.5), y / (wl * 0.5), seed + 1) + 0.15 * vnoise(x / (wl * 0.25), y / (wl * 0.25), seed + 2);
}

static inline double object_surface(const struct scene *s, double lx, double ly)
{
    switch (s->id) {
    case SC_REPEAT:
        /* A 2D grid of bumps with a period of 19.2 pixels at Z = 4: corners exist, so features do, and any whole number of periods of disparity looks the same. */
        return 128.0 + 55.0 * sin(2.0 * M_PI * lx / 0.12) + 55.0 * sin(2.0 * M_PI * ly / 0.12);
    case SC_FLAT:
        return 128.0 + 0.5 * sin(2.0 * M_PI * lx / 0.9);
    default:
        return texture(lx, ly, 0.10, 11);
    }
}

/* A ray through pixel (u, v) of camera `camera` at time t: the nearest surface's radiance. */
static inline double shade(const struct scene *s, const struct motion *m, double u, double v, int camera)
{
    const double rx = (u - s->cx) / s->f, ry = (v - s->cy) / s->f;
    const double ox = m->camX + (camera ? s->baseline : 0.0), oy = m->camY;
    if (m->occ_halfx > 0.0) {
        const double X = ox + rx * m->occ_z, Y = oy + ry * m->occ_z;
        if (fabs(X - m->occ_x) <= m->occ_halfx && fabs(Y - m->occ_y) <= m->occ_halfy)
            /* A thing that passes carries its surface with it; the bars of the older scenes do not, and show no motion but their edges. */
            return s->id == SC_PASSBY ? texture(X - m->occ_x, Y - m->occ_y, 0.06, 61) : 90.0 + 0.3 * texture(X, Y, 0.3, 21);
    }
    if (m->parts) {
        const double X = ox + rx * m->Z, Y = oy + ry * m->Z;
        for (int i = 0; i < m->parts; i++) {
            double lx = X - m->X - m->px[i], ly = Y - m->Y - m->py[i];
            const double c = cos(m->pspin[i]), sn = sin(m->pspin[i]), tx = c * lx + sn * ly, ty = c * ly - sn * lx;
            lx = tx;
            ly = ty;
            if (fabs(lx) <= 0.34 * m->halfx && fabs(ly) <= 0.5 * m->halfy)
                return texture(lx, ly, 0.06, 101 + 7 * (uint32_t)i);
        }
    } else {
        const double X = ox + rx * m->Z, Y = oy + ry * m->Z;
        double lx = X - m->X, ly = Y - m->Y;
        unturn(m, &lx, &ly);
        const int in = m->disc ? (lx * lx + ly * ly <= m->halfy * m->halfy) : (fabs(lx) <= m->halfx && fabs(ly) <= m->halfy);
        if (in)
            return object_surface(s, lx, ly);
    }
    if (s->id == SC_SHAPES) {
        const double Z2 = 5.0;
        const double X = ox + rx * Z2, Y = oy + ry * Z2;
        const double lx = X - m->second_x, ly = Y - 0.6;
        if (fabs(lx) <= 0.35 && fabs(ly) <= 0.25)
            return texture(lx, ly, 0.08, 41);
    }
    /* The background, 12 m away, still in the world. */
    const double Zb = 12.0;
    const double X = ox + rx * Zb, Y = oy + ry * Zb;
    if (busy_background(s))
        return texture(X, Y, 0.30, 31);
    return 40.0 + 0.6 * texture(X, Y, 0.45, 31);
}

/* SC_DUAL, camera 1: the ray of a pixel through a camera that is neither parallel to camera 0 nor the same lens. */
static inline double shade_b(const struct scene *s, const struct motion *m, double u, double v)
{
    const double bx = (u - s->cx) / s->b_f, by = (v - s->cy) / s->b_f;
    const double *R = s->b_R, *T = s->b_T;
    const double d[3] = {R[0] * bx + R[3] * by + R[6], R[1] * bx + R[4] * by + R[7], R[2] * bx + R[5] * by + R[8]};
    const double C[3] = {-(R[0] * T[0] + R[3] * T[1] + R[6] * T[2]), -(R[1] * T[0] + R[4] * T[1] + R[7] * T[2]), -(R[2] * T[0] + R[5] * T[1] + R[8] * T[2])};
    const double k = (m->Z - C[2]) / d[2];
    const double lx = C[0] + d[0] * k - m->X, ly = C[1] + d[1] * k - m->Y;
    if (k > 0.0 && fabs(lx) <= m->halfx && fabs(ly) <= m->halfy)
        return object_surface(s, lx, ly);
    const double kb = (12.0 - C[2]) / d[2];
    if (busy_background(s))
        return texture(C[0] + d[0] * kb, C[1] + d[1] * kb, 0.30, 31);
    return 40.0 + 0.6 * texture(C[0] + d[0] * kb, C[1] + d[1] * kb, 0.45, 31);
}

/* Normal noise from per-pixel hashes: a fixed result whatever the thread split. */
static inline double gauss_at(uint32_t a, uint32_t b)
{
    const double u1 = ((double)a + 0.5) / 4294967296.0, u2 = ((double)b + 0.5) / 4294967296.0;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

struct render_job {
    const struct scene *s;
    const struct motion *m;
    int frame, camera, y0, y1;
    uint8_t *img;
};

static void *render_rows(void *arg)
{
    const struct render_job *j = arg;
    const struct scene *s = j->s;
    const int n = s->samples;
    const double step = 1.0 / n, start = -0.5 + 0.5 * step;
    const int blurred = s->shutter > 0.0;
    /* The glitch: for ten frames the right image is wrong by 14 pixels. Geometry says one depth, the images another. */
    const double shift = (s->id == SC_GLITCH && j->camera == 1 && j->frame >= 70 && j->frame < 80) ? 14.0 : 0.0;
    for (int y = j->y0; y < j->y1; y++) {
        uint8_t *row = j->img + (size_t)y * (size_t)s->w;
        for (int x = 0; x < s->w; x++) {
            double sum = 0.0;
            for (int b = 0; b < n; b++)
                for (int a = 0; a < n; a++)
                    sum += (s->id == SC_DUAL || s->id == SC_DUALBOX) && j->camera == 1 ? shade_b(s, j->m, x + start + a * step, y + start + b * step) : shade(s, blurred ? &j->m[a * n + b] : j->m, x + start + a * step + shift, y + start + b * step, j->camera);
            double v = sum / (n * n);
            v = v * j->m->gain + j->m->offset;
            if (j->m->noise > 0.0)
                v += (s->clean ? 0.0 : j->m->noise) * gauss_at(hash2(x, y, (uint32_t)(j->frame * 2 + j->camera)), hash2(y, x, (uint32_t)(j->frame * 2 + j->camera + 7777)));
            v = floor(v + 0.5);
            row[x] = (uint8_t)(v < 0.0 ? 0 : v > 255.0 ? 255 : v);
        }
    }
    return NULL;
}

#define POOL 14

void scene_render(const struct scene *s, int frame, int camera, uint8_t *img) { scene_render_at(s, (double)frame / s->fps + ((s->id == SC_DUAL || s->id == SC_DUALBOX) && camera == 1 ? s->b_delay_s : 0.0), frame, camera, img); }

void scene_render_at(const struct scene *s, double seconds, int frame, int camera, uint8_t *img)
{
    /* A late shutter sees the scene later; a long one sees all of its exposure, one instant to a sample, centred on the frame's own time. */
    static __thread struct motion m[16];
    const double when = seconds;
    motion_at(s, when, &m[0]);
    if (s->shutter > 0.0) {
        const int all = s->samples * s->samples;
        for (int i = 0; i < all; i++)
            motion_at(s, when + s->shutter / s->fps * ((i + 0.5) / all - 0.5), &m[i]);
    }
    pthread_t th[POOL];
    struct render_job jobs[POOL];
    for (int i = 0; i < POOL; i++) {
        jobs[i] = (struct render_job){s, m, frame, camera, s->h * i / POOL, s->h * (i + 1) / POOL, img};
        pthread_create(&th[i], NULL, render_rows, &jobs[i]);
    }
    for (int i = 0; i < POOL; i++)
        pthread_join(th[i], NULL);
}

/* The fraction of the object's area that is really seen, by sampling it. */
static double visible_fraction(const struct scene *s, const struct motion *m)
{
    int seen = 0, total = 0;
    for (int b = 0; b < 21; b++)
        for (int a = 0; a < 21; a++) {
            const double lx = m->halfx * (-1.0 + a / 10.0), ly = m->halfy * (-1.0 + b / 10.0);
            if (m->disc && lx * lx + ly * ly > m->halfy * m->halfy)
                continue;
            total++;
            const double X = m->X + lx, Y = m->Y + ly;
            const double u = s->f * (X - m->camX) / m->Z + s->cx, v = s->f * (Y - m->camY) / m->Z + s->cy;
            if (u < 0.0 || v < 0.0 || u >= s->w || v >= s->h)
                continue;
            if (m->occ_halfx > 0.0) {
                const double ox = m->camX + (u - s->cx) / s->f * m->occ_z, oy = m->camY + (v - s->cy) / s->f * m->occ_z;
                if (fabs(ox - m->occ_x) <= m->occ_halfx && fabs(oy - m->occ_y) <= m->occ_halfy)
                    continue;
            }
            seen++;
        }
    return total ? (double)seen / total : 0.0;
}

void scene_truth(const struct scene *s, int frame, struct truth *t) { scene_truth_at(s, (double)frame / s->fps, t); }

void scene_truth_at(const struct scene *s, double seconds, struct truth *t)
{
    struct motion m;
    motion_at(s, seconds, &m);
    /* In the moving left camera's frame: X relative to the camera, velocities relative too, because the camera is what measures. */
    t->X = m.X - m.camX;
    t->Y = m.Y - m.camY;
    t->Z = m.Z;
    t->Vx = m.Vx;
    t->Vy = m.Vy;
    t->Vz = m.Vz;
    t->u = s->f * t->X / m.Z + s->cx;
    t->v = s->f * t->Y / m.Z + s->cy;
    t->ur = s->f * (t->X - s->baseline) / m.Z + s->cx;
    t->on_screen = t->u >= 0.0 && t->v >= 0.0 && t->u < s->w && t->v < s->h;
    t->visible = visible_fraction(s, &m);
    t->half_w = s->f * m.halfx / m.Z;
    t->half_h = s->f * m.halfy / m.Z;
    if (s->id == SC_DUAL || s->id == SC_DUALBOX) {
        const double *R = s->b_R, *T = s->b_T;
        const double q[3] = {R[0] * t->X + R[1] * t->Y + R[2] * t->Z + T[0], R[3] * t->X + R[4] * t->Y + R[5] * t->Z + T[1], R[6] * t->X + R[7] * t->Y + R[8] * t->Z + T[2]};
        t->ub = s->b_f * q[0] / q[2] + s->cx;
        t->vb = s->b_f * q[1] / q[2] + s->cy;
        t->half_wb = s->b_f * m.halfx / q[2];
        t->half_hb = s->b_f * m.halfy / q[2];
    }
}

int scene_map_left(const struct scene *s, double t0, double x0, double y0, double t1, double *x1, double *y1)
{
    struct motion a, b;
    motion_at(s, t0, &a);
    motion_at(s, t1, &b);
    /* A thing of parts has no one motion to carry a point by. */
    if (a.parts)
        return 0;
    const double rx = (x0 - s->cx) / s->f, ry = (y0 - s->cy) / s->f;
    /* The nearest surface decides what the point is. */
    if (a.occ_halfx > 0.0) {
        const double X = a.camX + rx * a.occ_z, Y = a.camY + ry * a.occ_z;
        if (fabs(X - a.occ_x) <= a.occ_halfx && fabs(Y - a.occ_y) <= a.occ_halfy)
            return 0;
    }
    const double X = a.camX + rx * a.Z, Y = a.camY + ry * a.Z;
    double lx = X - a.X, ly = Y - a.Y;
    {
        double ox = lx, oy = ly;
        unturn(&a, &ox, &oy);
        const int in = a.disc ? (ox * ox + oy * oy <= a.halfy * a.halfy) : (fabs(ox) <= a.halfx && fabs(oy) <= a.halfy);
        if (!in)
            return 0;
    }
    if (b.spin != a.spin) {
        const double c = cos(b.spin - a.spin), sn = sin(b.spin - a.spin), px = lx, py = ly;
        lx = c * px - sn * py;
        ly = sn * px + c * py;
    }
    const double Xb = b.X + lx - b.camX, Yb = b.Y + ly - b.camY;
    *x1 = s->f * Xb / b.Z + s->cx;
    *y1 = s->f * Yb / b.Z + s->cy;
    if (b.occ_halfx > 0.0) {
        const double ox = b.camX + (*x1 - s->cx) / s->f * b.occ_z, oy = b.camY + (*y1 - s->cy) / s->f * b.occ_z;
        if (fabs(ox - b.occ_x) <= b.occ_halfx && fabs(oy - b.occ_y) <= b.occ_halfy)
            return 0;
    }
    return 1;
}
