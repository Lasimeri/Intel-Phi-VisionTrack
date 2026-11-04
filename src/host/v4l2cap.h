/* v4l2cap.h: a V4L2 capture device, memory-mapped, MJPEG, decoded to its luma plane: the image the card receives. The acquisition time is the kernel's own buffer timestamp. */
#ifndef VT_V4L2CAP_H
#define VT_V4L2CAP_H

#include <stdint.h>

#define V4L2CAP_PATH 256

struct v4l2cap;

/* Which device is which camera, from a local file of lines "L PATH", "R PATH", "D PATH"; # starts a comment. The file stays out of the repository, because a by-id path carries the camera's serial
 * number. Fills names[0..2] for L, R, D where the file names them and leaves the others as they were. How many it set, or -1 when the file cannot be opened. */
int v4l2cap_names(const char *path, char names[3][V4L2CAP_PATH]);

/* Opens the device at w x h, fps; NULL with a message when it refuses. raw 0: MJPEG. raw 1: YUYV, the luma taken directly: its data rate is exact, so a camera that claims its whole
 * USB allowance for MJPEG at any frame rate claims only what it needs. */
struct v4l2cap *v4l2cap_open(const char *dev, int w, int h, int fps, int raw);
/* The next frame's luma into gray (w x h bytes, pitch w). 1 with acq_ns set, CLOCK_MONOTONIC; 0 when none came within timeout_ms; -1 on a device error. */
int v4l2cap_next(struct v4l2cap *c, uint8_t *gray, uint64_t *acq_ns, int timeout_ms);
void v4l2cap_close(struct v4l2cap *c);
/* The device's frames that failed to decode, since opening. */
uint64_t v4l2cap_bad(const struct v4l2cap *c);
/* Exposure may not stretch a frame past its time: the rate asked for holds in the dark, at the cost of light. 0, or -1 where the device has no such control. */
int v4l2cap_steady_rate(struct v4l2cap *c);

#endif
