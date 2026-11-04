/* v4l2cap.c: see v4l2cap.h. Only the luma of the JPEG is decoded: no chroma work, and the Y plane is exactly the monochrome image the brief asks for. */
#define _GNU_SOURCE
#include "v4l2cap.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <turbojpeg.h>
#include <unistd.h>

#define NBUF 4

struct v4l2cap {
    int fd, w, h;
    void *map[NBUF];
    size_t len[NBUF];
    tjhandle tj;
    uint64_t bad;
    int raw;
    uint32_t bytesperline;
};

int v4l2cap_names(const char *path, char names[3][V4L2CAP_PATH])
{
    FILE *f = fopen(path, "r");
    if (!f)
        return -1;
    static const char order[] = "LRD";
    char line[V4L2CAP_PATH + 16], tag[3], dev[V4L2CAP_PATH];
    unsigned seen = 0;
    while (fgets(line, sizeof line, f)) {
        /* Two characters read for the tag, so "Left /dev/x" is refused instead of naming the device "eft". */
        if (sscanf(line, "%2s %255s", tag, dev) != 2 || tag[1])
            continue;
        const char *const at = strchr(order, tag[0]);
        if (at) {
            const int k = (int)(at - order);
            strcpy(names[k], dev);
            seen |= 1u << k;
        }
    }
    fclose(f);
    return __builtin_popcount(seen);
}

static int xioctl(int fd, unsigned long req, void *arg)
{
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r < 0 && errno == EINTR);
    return r;
}

/* Every way out of an open that did not finish. A descriptor left holding buffers makes the device answer EBUSY to every later attempt, in this process and any other: the first failure's reason would be lost behind it. */
static struct v4l2cap *open_failed(struct v4l2cap *c)
{
    for (int i = 0; i < NBUF; i++)
        if (c->map[i])
            munmap(c->map[i], c->len[i]);
    close(c->fd);
    free(c);
    return NULL;
}

struct v4l2cap *v4l2cap_open(const char *dev, int w, int h, int fps, int raw)
{
    struct v4l2cap *c = calloc(1, sizeof *c);
    const uint32_t want = raw ? V4L2_PIX_FMT_YUYV : V4L2_PIX_FMT_MJPEG;
    const char *want_name = raw ? "YUYV" : "MJPEG";
    c->w = w;
    c->h = h;
    c->raw = raw;
    c->fd = open(dev, O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (c->fd < 0) {
        fprintf(stderr, "v4l2cap: %s: %s\n", dev, strerror(errno));
        free(c);
        return NULL;
    }
    struct v4l2_format f;
    memset(&f, 0, sizeof f);
    f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    f.fmt.pix.width = (uint32_t)w;
    f.fmt.pix.height = (uint32_t)h;
    f.fmt.pix.pixelformat = want;
    f.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(c->fd, VIDIOC_S_FMT, &f) || (int)f.fmt.pix.width != w || (int)f.fmt.pix.height != h || f.fmt.pix.pixelformat != want) {
        fprintf(stderr, "v4l2cap: %s refuses %s %dx%d (%s)\n", dev, want_name, w, h, strerror(errno));
        return open_failed(c);
    }
    c->bytesperline = f.fmt.pix.bytesperline ? f.fmt.pix.bytesperline : (uint32_t)w * 2;
    struct v4l2_streamparm p;
    memset(&p, 0, sizeof p);
    p.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    p.parm.capture.timeperframe.numerator = 1;
    p.parm.capture.timeperframe.denominator = (uint32_t)fps;
    xioctl(c->fd, VIDIOC_S_PARM, &p);
    struct v4l2_requestbuffers rb;
    memset(&rb, 0, sizeof rb);
    rb.count = NBUF;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_MMAP;
    if (xioctl(c->fd, VIDIOC_REQBUFS, &rb) || rb.count < NBUF) {
        fprintf(stderr, "v4l2cap: %s: no %d buffers (%s)\n", dev, NBUF, strerror(errno));
        return open_failed(c);
    }
    for (unsigned i = 0; i < NBUF; i++) {
        struct v4l2_buffer b;
        memset(&b, 0, sizeof b);
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index = i;
        if (xioctl(c->fd, VIDIOC_QUERYBUF, &b)) {
            fprintf(stderr, "v4l2cap: %s: buffer %u: %s\n", dev, i, strerror(errno));
            return open_failed(c);
        }
        c->len[i] = b.length;
        c->map[i] = mmap(NULL, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, c->fd, (off_t)b.m.offset);
        if (c->map[i] == MAP_FAILED) {
            c->map[i] = NULL;
            fprintf(stderr, "v4l2cap: %s: map %u: %s\n", dev, i, strerror(errno));
            return open_failed(c);
        }
        if (xioctl(c->fd, VIDIOC_QBUF, &b)) {
            fprintf(stderr, "v4l2cap: %s: queue %u: %s\n", dev, i, strerror(errno));
            return open_failed(c);
        }
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(c->fd, VIDIOC_STREAMON, &type)) {
        /* ENOSPC is the USB bus: the cameras on it have together claimed more isochronous bandwidth than it has. */
        fprintf(stderr, "v4l2cap: %s: stream on: %s\n", dev, strerror(errno));
        return open_failed(c);
    }
    c->tj = tjInitDecompress();
    return c;
}

int v4l2cap_next(struct v4l2cap *c, uint8_t *gray, uint64_t *acq_ns, int timeout_ms)
{
    struct pollfd pf = {.fd = c->fd, .events = POLLIN};
    const int r = poll(&pf, 1, timeout_ms);
    if (r == 0)
        return 0;
    if (r < 0)
        return errno == EINTR ? 0 : -1;
    struct v4l2_buffer b;
    memset(&b, 0, sizeof b);
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    if (xioctl(c->fd, VIDIOC_DQBUF, &b))
        return errno == EAGAIN ? 0 : -1;
    /* The kernel's timestamp: taken when the frame was finished arriving, on CLOCK_MONOTONIC for UVC. */
    *acq_ns = (uint64_t)b.timestamp.tv_sec * 1000000000ull + (uint64_t)b.timestamp.tv_usec * 1000ull;
    int ok = 1;
    if (c->raw) {
        /* Y0 U Y1 V: the luma is every other byte. A frame the transfer tore is a frame not measured. */
        if ((b.flags & V4L2_BUF_FLAG_ERROR) || b.bytesused < c->bytesperline * (uint32_t)c->h) {
            c->bad++;
            ok = 0;
        } else {
            const uint8_t *src = c->map[b.index];
            for (int y = 0; y < c->h; y++) {
                const uint8_t *s = src + (size_t)y * c->bytesperline;
                uint8_t *d = gray + (size_t)y * (size_t)c->w;
                for (int x = 0; x < c->w; x++)
                    d[x] = s[2 * x];
            }
        }
    } else if (tjDecompress2(c->tj, c->map[b.index], b.bytesused, gray, c->w, c->w, c->h, TJPF_GRAY, 0)) {
        /* A warning still decodes the frame; an error does not. */
        if (tjGetErrorCode(c->tj) != TJERR_WARNING) {
            c->bad++;
            ok = 0;
        }
    }
    xioctl(c->fd, VIDIOC_QBUF, &b);
    return ok;
}

uint64_t v4l2cap_bad(const struct v4l2cap *c) { return c->bad; }

void v4l2cap_close(struct v4l2cap *c)
{
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(c->fd, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < NBUF; i++)
        munmap(c->map[i], c->len[i]);
    tjDestroy(c->tj);
    close(c->fd);
    free(c);
}

int v4l2cap_steady_rate(struct v4l2cap *c)
{
    struct v4l2_control ctl;
    memset(&ctl, 0, sizeof ctl);
    ctl.id = V4L2_CID_EXPOSURE_AUTO_PRIORITY;
    ctl.value = 0;
    return ioctl(c->fd, VIDIOC_S_CTRL, &ctl);
}
