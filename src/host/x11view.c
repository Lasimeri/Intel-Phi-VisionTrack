/* x11view.c: see x11view.h. */
#define _GNU_SOURCE
#include "x11view.h"

#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>

struct priv {
    Display *d;
    Window win;
    GC gc;
    XImage *img;
    XShmSegmentInfo shm;
    Atom wm_delete;
};

int x11view_open(struct x11view *v, int w, int h, int x, int y, const char *title)
{
    struct priv *p = calloc(1, sizeof *p);
    p->d = XOpenDisplay(NULL);
    if (!p->d) {
        fprintf(stderr, "x11view: no display (DISPLAY=%s)\n", getenv("DISPLAY") ? getenv("DISPLAY") : "");
        return -1;
    }
    if (!XShmQueryExtension(p->d)) {
        fprintf(stderr, "x11view: no MIT-SHM\n");
        return -1;
    }
    const int scr = DefaultScreen(p->d);
    Visual *vis = DefaultVisual(p->d, scr);
    p->win = XCreateSimpleWindow(p->d, RootWindow(p->d, scr), x < 0 ? 0 : x, y < 0 ? 0 : y, (unsigned)w, (unsigned)h, 0, 0, 0);
    XStoreName(p->d, p->win, title);
    XSelectInput(p->d, p->win, KeyPressMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask | StructureNotifyMask);
    p->wm_delete = XInternAtom(p->d, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(p->d, p->win, &p->wm_delete, 1);
    /* Not resizable, placed where asked, and no focus taken on mapping: a restart must not pull the keyboard from what the user is doing. */
    XSizeHints sh;
    memset(&sh, 0, sizeof sh);
    sh.flags = PMinSize | PMaxSize | (x >= 0 ? (PPosition | USPosition) : 0);
    sh.x = x;
    sh.y = y;
    sh.min_width = sh.max_width = w;
    sh.min_height = sh.max_height = h;
    XSetWMNormalHints(p->d, p->win, &sh);
    const unsigned long no_time = 0;
    XChangeProperty(p->d, p->win, XInternAtom(p->d, "_NET_WM_USER_TIME", False), XA_CARDINAL, 32, PropModeReplace, (const unsigned char *)&no_time, 1);
    p->gc = XCreateGC(p->d, p->win, 0, NULL);
    p->img = XShmCreateImage(p->d, vis, (unsigned)DefaultDepth(p->d, scr), ZPixmap, NULL, &p->shm, (unsigned)w, (unsigned)h);
    if (!p->img || p->img->bits_per_pixel != 32) {
        fprintf(stderr, "x11view: this display has no 32-bit image format\n");
        return -1;
    }
    p->shm.shmid = shmget(IPC_PRIVATE, (size_t)p->img->bytes_per_line * (size_t)h, IPC_CREAT | 0600);
    p->shm.shmaddr = p->img->data = shmat(p->shm.shmid, NULL, 0);
    p->shm.readOnly = False;
    XShmAttach(p->d, &p->shm);
    XSync(p->d, False);
    shmctl(p->shm.shmid, IPC_RMID, NULL);
    XMapWindow(p->d, p->win);
    v->w = w;
    v->h = h;
    v->px = (uint32_t *)p->img->data;
    v->priv = p;
    memset(v->px, 0, (size_t)p->img->bytes_per_line * (size_t)h);
    return 0;
}

int x11view_poll(struct x11view *v, struct x11event *e)
{
    struct priv *p = v->priv;
    memset(e, 0, sizeof *e);
    while (XPending(p->d)) {
        XEvent x;
        XNextEvent(p->d, &x);
        switch (x.type) {
        case KeyPress: {
            char buf[8];
            KeySym ks;
            if (XLookupString(&x.xkey, buf, sizeof buf, &ks, NULL) >= 1) {
                e->type = X11_KEY;
                e->key = buf[0];
                return e->type;
            }
            break;
        }
        case ButtonPress:
            e->type = X11_PRESS;
            e->x = x.xbutton.x;
            e->y = x.xbutton.y;
            e->button = (int)x.xbutton.button;
            return e->type;
        case ButtonRelease:
            e->type = X11_RELEASE;
            e->x = x.xbutton.x;
            e->y = x.xbutton.y;
            e->button = (int)x.xbutton.button;
            return e->type;
        case MotionNotify:
            e->type = X11_MOTION;
            e->x = x.xmotion.x;
            e->y = x.xmotion.y;
            return e->type;
        case ClientMessage:
            if ((Atom)x.xclient.data.l[0] == p->wm_delete) {
                e->type = X11_QUIT;
                return e->type;
            }
            break;
        default:
            break;
        }
    }
    return X11_NONE;
}

void x11view_present(struct x11view *v)
{
    struct priv *p = v->priv;
    XShmPutImage(p->d, p->win, p->gc, p->img, 0, 0, 0, 0, (unsigned)v->w, (unsigned)v->h, False);
    XSync(p->d, False);
}

int x11view_where(struct x11view *v, int *x, int *y)
{
    struct priv *p = v->priv;
    Window child;
    int ax, ay;
    if (!XTranslateCoordinates(p->d, p->win, DefaultRootWindow(p->d), 0, 0, &ax, &ay, &child))
        return -1;
    /* _NET_FRAME_EXTENTS: left, right, top, bottom, from the window manager. */
    Atom type;
    int format;
    unsigned long n, after;
    unsigned char *data = NULL;
    long left = 0, top = 0;
    if (XGetWindowProperty(p->d, p->win, XInternAtom(p->d, "_NET_FRAME_EXTENTS", False), 0, 4, False, XA_CARDINAL, &type, &format, &n, &after, &data) == Success && data) {
        if (n == 4) {
            const long *e = (const long *)data;
            left = e[0];
            top = e[2];
        }
        XFree(data);
    }
    *x = ax - (int)left;
    *y = ay - (int)top;
    return 0;
}

void x11view_close(struct x11view *v)
{
    struct priv *p = v->priv;
    XShmDetach(p->d, &p->shm);
    shmdt(p->shm.shmaddr);
    XCloseDisplay(p->d);
    free(p);
    v->priv = NULL;
}

void x11view_gray(struct x11view *v, const uint8_t *g, int gw, int gh, int pitch, int x, int y, int step)
{
    for (int r = 0; r * step < gh && y + r < v->h; r++) {
        uint32_t *d = v->px + (size_t)(y + r) * (size_t)v->w + x;
        const uint8_t *s = g + (size_t)(r * step) * (size_t)pitch;
        const int n = gw / step < v->w - x ? gw / step : v->w - x;
        for (int c = 0; c < n; c++)
            d[c] = (uint32_t)s[c * step] * 0x010101u;
    }
}

void x11view_gray_mirror(struct x11view *v, const uint8_t *g, int gw, int gh, int pitch, int x, int y)
{
    for (int r = 0; r < gh && y + r < v->h; r++) {
        uint32_t *d = v->px + (size_t)(y + r) * (size_t)v->w + x;
        const uint8_t *s = g + (size_t)r * (size_t)pitch;
        const int n = gw < v->w - x ? gw : v->w - x;
        for (int c = 0; c < n; c++)
            d[c] = (uint32_t)s[gw - 1 - c] * 0x010101u;
    }
}

void x11view_fill(struct x11view *v, int x, int y, int w, int h, uint32_t c)
{
    for (int r = y; r < y + h && r < v->h; r++) {
        if (r < 0)
            continue;
        for (int k = x < 0 ? 0 : x; k < x + w && k < v->w; k++)
            v->px[(size_t)r * (size_t)v->w + (size_t)k] = c;
    }
}

static inline void put(struct x11view *v, int x, int y, uint32_t c)
{
    if (x >= 0 && y >= 0 && x < v->w && y < v->h)
        v->px[(size_t)y * (size_t)v->w + (size_t)x] = c;
}

void x11view_dash(struct x11view *v, int x0, int y0, int x1, int y1, uint32_t c, int on, int off)
{
    const int dx = abs(x1 - x0), dy = -abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int e = dx + dy, n = 0;
    const int period = on + off;
    for (int guard = 0; guard < 20000; guard++) {
        if (!off || n % period < on)
            put(v, x0, y0, c);
        n++;
        if (x0 == x1 && y0 == y1)
            break;
        const int e2 = 2 * e;
        if (e2 >= dy) {
            e += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            e += dx;
            y0 += sy;
        }
    }
}

void x11view_line(struct x11view *v, int x0, int y0, int x1, int y1, uint32_t c) { x11view_dash(v, x0, y0, x1, y1, c, 1, 0); }

static const uint8_t font5x7[95][5] = {
    {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x5F, 0x00, 0x00}, {0x00, 0x07, 0x00, 0x07, 0x00}, {0x14, 0x7F, 0x14, 0x7F, 0x14}, {0x24, 0x2A, 0x7F, 0x2A, 0x12},
    {0x23, 0x13, 0x08, 0x64, 0x62}, {0x36, 0x49, 0x56, 0x20, 0x50}, {0x00, 0x08, 0x07, 0x03, 0x00}, {0x00, 0x1C, 0x22, 0x41, 0x00}, {0x00, 0x41, 0x22, 0x1C, 0x00},
    {0x2A, 0x1C, 0x7F, 0x1C, 0x2A}, {0x08, 0x08, 0x3E, 0x08, 0x08}, {0x00, 0x80, 0x70, 0x30, 0x00}, {0x08, 0x08, 0x08, 0x08, 0x08}, {0x00, 0x00, 0x60, 0x60, 0x00},
    {0x20, 0x10, 0x08, 0x04, 0x02}, {0x3E, 0x51, 0x49, 0x45, 0x3E}, {0x00, 0x42, 0x7F, 0x40, 0x00}, {0x72, 0x49, 0x49, 0x49, 0x46}, {0x21, 0x41, 0x49, 0x4D, 0x33},
    {0x18, 0x14, 0x12, 0x7F, 0x10}, {0x27, 0x45, 0x45, 0x45, 0x39}, {0x3C, 0x4A, 0x49, 0x49, 0x31}, {0x41, 0x21, 0x11, 0x09, 0x07}, {0x36, 0x49, 0x49, 0x49, 0x36},
    {0x46, 0x49, 0x49, 0x29, 0x1E}, {0x00, 0x00, 0x14, 0x00, 0x00}, {0x00, 0x40, 0x34, 0x00, 0x00}, {0x00, 0x08, 0x14, 0x22, 0x41}, {0x14, 0x14, 0x14, 0x14, 0x14},
    {0x00, 0x41, 0x22, 0x14, 0x08}, {0x02, 0x01, 0x59, 0x09, 0x06}, {0x3E, 0x41, 0x5D, 0x59, 0x4E}, {0x7C, 0x12, 0x11, 0x12, 0x7C}, {0x7F, 0x49, 0x49, 0x49, 0x36},
    {0x3E, 0x41, 0x41, 0x41, 0x22}, {0x7F, 0x41, 0x41, 0x41, 0x3E}, {0x7F, 0x49, 0x49, 0x49, 0x41}, {0x7F, 0x09, 0x09, 0x09, 0x01}, {0x3E, 0x41, 0x41, 0x51, 0x73},
    {0x7F, 0x08, 0x08, 0x08, 0x7F}, {0x00, 0x41, 0x7F, 0x41, 0x00}, {0x20, 0x40, 0x41, 0x3F, 0x01}, {0x7F, 0x08, 0x14, 0x22, 0x41}, {0x7F, 0x40, 0x40, 0x40, 0x40},
    {0x7F, 0x02, 0x1C, 0x02, 0x7F}, {0x7F, 0x04, 0x08, 0x10, 0x7F}, {0x3E, 0x41, 0x41, 0x41, 0x3E}, {0x7F, 0x09, 0x09, 0x09, 0x06}, {0x3E, 0x41, 0x51, 0x21, 0x5E},
    {0x7F, 0x09, 0x19, 0x29, 0x46}, {0x26, 0x49, 0x49, 0x49, 0x32}, {0x03, 0x01, 0x7F, 0x01, 0x03}, {0x3F, 0x40, 0x40, 0x40, 0x3F}, {0x1F, 0x20, 0x40, 0x20, 0x1F},
    {0x3F, 0x40, 0x38, 0x40, 0x3F}, {0x63, 0x14, 0x08, 0x14, 0x63}, {0x03, 0x04, 0x78, 0x04, 0x03}, {0x61, 0x59, 0x49, 0x4D, 0x43}, {0x00, 0x7F, 0x41, 0x41, 0x41},
    {0x02, 0x04, 0x08, 0x10, 0x20}, {0x00, 0x41, 0x41, 0x41, 0x7F}, {0x04, 0x02, 0x01, 0x02, 0x04}, {0x40, 0x40, 0x40, 0x40, 0x40}, {0x00, 0x03, 0x07, 0x08, 0x00},
    {0x20, 0x54, 0x54, 0x78, 0x40}, {0x7F, 0x28, 0x44, 0x44, 0x38}, {0x38, 0x44, 0x44, 0x44, 0x28}, {0x38, 0x44, 0x44, 0x28, 0x7F}, {0x38, 0x54, 0x54, 0x54, 0x18},
    {0x00, 0x08, 0x7E, 0x09, 0x02}, {0x18, 0xA4, 0xA4, 0x9C, 0x78}, {0x7F, 0x08, 0x04, 0x04, 0x78}, {0x00, 0x44, 0x7D, 0x40, 0x00}, {0x20, 0x40, 0x40, 0x3D, 0x00},
    {0x7F, 0x10, 0x28, 0x44, 0x00}, {0x00, 0x41, 0x7F, 0x40, 0x00}, {0x7C, 0x04, 0x78, 0x04, 0x78}, {0x7C, 0x08, 0x04, 0x04, 0x78}, {0x38, 0x44, 0x44, 0x44, 0x38},
    {0xFC, 0x18, 0x24, 0x24, 0x18}, {0x18, 0x24, 0x24, 0x18, 0xFC}, {0x7C, 0x08, 0x04, 0x04, 0x08}, {0x48, 0x54, 0x54, 0x54, 0x24}, {0x04, 0x04, 0x3F, 0x44, 0x24},
    {0x3C, 0x40, 0x40, 0x20, 0x7C}, {0x1C, 0x20, 0x40, 0x20, 0x1C}, {0x3C, 0x40, 0x30, 0x40, 0x3C}, {0x44, 0x28, 0x10, 0x28, 0x44}, {0x4C, 0x90, 0x90, 0x90, 0x7C},
    {0x44, 0x64, 0x54, 0x4C, 0x44}, {0x00, 0x08, 0x36, 0x41, 0x00}, {0x00, 0x00, 0x77, 0x00, 0x00}, {0x00, 0x41, 0x36, 0x08, 0x00}, {0x02, 0x01, 0x02, 0x04, 0x02},
};

void x11view_text(struct x11view *v, int x, int y, int s, uint32_t c, const char *t)
{
    for (; *t; t++, x += 6 * s) {
        const unsigned ch = (unsigned char)*t;
        if (ch < 32 || ch > 126)
            continue;
        const uint8_t *g = font5x7[ch - 32];
        for (int col = 0; col < 5; col++)
            for (int row = 0; row < 8; row++)
                if (g[col] >> row & 1)
                    x11view_fill(v, x + col * s, y + row * s, s, s, c);
    }
}
