/* x11view.h: one window, one XSHM image of 32-bit pixels, and the few drawing operations a monochrome display needs. No toolkit. */
#ifndef VT_X11VIEW_H
#define VT_X11VIEW_H

#include <stdint.h>

struct x11view {
    int w, h;
    uint32_t *px;      /* w x h, 0x00RRGGBB */
    void *priv;
};

enum { X11_NONE, X11_QUIT, X11_KEY, X11_PRESS, X11_MOTION, X11_RELEASE };
struct x11event {
    int type, x, y, button;
    char key;
};

/* A window of w x h at (x, y) on the desktop (negative: where the window manager likes). It does not take the keyboard from whatever has it. 0, or -1 with a message. */
int x11view_open(struct x11view *v, int w, int h, int x, int y, const char *title);
/* One pending event, or X11_NONE. */
int x11view_poll(struct x11view *v, struct x11event *e);
/* Shows the image. */
void x11view_present(struct x11view *v);
/* Queues one part of the image to be shown; x11view_sync then waits until the server has all of them. */
void x11view_present_rect(struct x11view *v, int x, int y, int w, int h);
void x11view_sync(struct x11view *v);
void x11view_close(struct x11view *v);
/* Where the window's frame is on the desktop, title bar included: what to ask for to come back to the same place. 0, or -1. */
int x11view_where(struct x11view *v, int *x, int *y);

/* Grey frame at (x, y), scale 1, or every `step`-th pixel when smaller. */
void x11view_gray(struct x11view *v, const uint8_t *gray, int gw, int gh, int pitch, int x, int y, int step);
/* The same at scale 1, left and right exchanged. */
void x11view_gray_mirror(struct x11view *v, const uint8_t *gray, int gw, int gh, int pitch, int x, int y);
void x11view_fill(struct x11view *v, int x, int y, int w, int h, uint32_t c);
void x11view_line(struct x11view *v, int x0, int y0, int x1, int y1, uint32_t c);
/* A line of dashes: how a prediction is told from a measurement, in any light. */
void x11view_dash(struct x11view *v, int x0, int y0, int x1, int y1, uint32_t c, int on, int off);
/* ASCII, 5 x 7 cells at scale s: a 6 s by 8 s advance. */
void x11view_text(struct x11view *v, int x, int y, int s, uint32_t c, const char *t);

#endif
