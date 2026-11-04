/* vt_proto.h: the contract between the host and the card program, in one part of the shared host-memory region.
 *
 * The region is /dev/shm/phi-hostmem on the host (write-back) and /dev/phihost on the card (uncached: a card read is a PCIe round trip, so the card polls one word, never
 * a table). Frames and results move by the card's own DMA channel. Words the card writes for the host are written by DMA status descriptors, after the data they announce on
 * the same channel, or by a direct uncached store where nothing precedes them.
 *
 * Traffic is message objects, after the peer-vector machine of US7418574B2 (docs/architecture.md): a typed header naming the destination, priority, algorithm, length and
 * the unique instance, on messages both ways, and carrying data, commands, or both. A malformed one is dropped and reported, never guessed at.
 *
 * Offsets taken by other projects are listed in docs/compatibility.md. */
#ifndef VT_PROTO_H
#define VT_PROTO_H

#include <stdint.h>

#define VT_PART_OFFSET (1536ull << 20)
#define VT_PART_BYTES (256ull << 20)
#define VT_MAGIC 0x3154435254495456ull /* "VTIRTCT1" */

#define VT_CTL_OFFSET 0ull              /* the control area */
#define VT_RES_OFFSET (1ull << 20)      /* the result ring */
#define VT_RES_SLOTS 16
#define VT_RES_BYTES (64u << 10)
#define VT_DEPTH_OFFSET (2ull << 20)    /* the depth map: two buffers, depth_pub names the one that is complete */
#define VT_DEPTH_BYTES (2ull << 20)     /* the map and every camera's cells: at native flow, 16,200 cells each for L and R */
#define VT_XFER_OFFSET (6ull << 20)     /* the transfer benchmark's buffer, up to 12 MiB */
#define VT_XFER_BYTES (12ull << 20)
#define VT_FRAME_OFFSET (18ull << 20)   /* frame slots */
#define VT_MAX_SLOTS 16
/* Generated pictures, 8-bit grey, at the top of the part: a camera's picture is whole for three more of its generated frames, which is longer than any picture waits to be shown. */
#define VT_GEN_CAMERAS 3
#define VT_GEN_SLOTS 4
#define VT_GEN_PIC_BYTES (4ull << 20)
#define VT_GEN_BYTES (VT_GEN_CAMERAS * VT_GEN_SLOTS * VT_GEN_PIC_BYTES)
#define VT_GEN_OFFSET (VT_PART_BYTES - VT_GEN_BYTES)
static inline uint64_t vt_gen_offset(uint32_t camera, uint32_t slot) { return VT_GEN_OFFSET + ((uint64_t)camera * VT_GEN_SLOTS + slot) * VT_GEN_PIC_BYTES; }
/* Pictures from FSR, one camera, under the generated ones: two a real frame, held until their moment comes. */
#define VT_FSR_SLOTS 8
#define VT_FSR_PIC_BYTES (4ull << 20)
#define VT_FSR_BYTES (VT_FSR_SLOTS * VT_FSR_PIC_BYTES)
#define VT_FSR_OFFSET (VT_GEN_OFFSET - VT_FSR_BYTES)
static inline uint64_t vt_fsr_offset(uint32_t slot) { return VT_FSR_OFFSET + (uint64_t)slot * VT_FSR_PIC_BYTES; }
/* The orange vectors over those pictures: one plane a real frame, brought down with them, paired by instance when the picture is drawn. */
#define VT_VEC_SLOTS 8
#define VT_VEC_PIC_BYTES (2ull << 20)
#define VT_VEC_BYTES (VT_VEC_SLOTS * VT_VEC_PIC_BYTES)
#define VT_VEC_OFFSET (VT_FSR_OFFSET - VT_VEC_BYTES)
static inline uint64_t vt_vec_offset(uint32_t slot) { return VT_VEC_OFFSET + (uint64_t)slot * VT_VEC_PIC_BYTES; }
/* L and R's vectors, a frame at a time, apart from the map: found by instant when a picture comes due. */
#define VT_CELL_CAMERAS 2
#define VT_CELL_SLOTS 8
#define VT_CELL_SLOT_BYTES (512ull << 10)
#define VT_CELLS_BYTES (VT_CELL_CAMERAS * VT_CELL_SLOTS * VT_CELL_SLOT_BYTES)
#define VT_CELLS_OFFSET (VT_VEC_OFFSET - VT_CELLS_BYTES)
static inline uint64_t vt_cells_offset(uint32_t camera, uint32_t slot) { return VT_CELLS_OFFSET + ((uint64_t)camera * VT_CELL_SLOTS + slot) * VT_CELL_SLOT_BYTES; }
#define VT_FRAME_AREA (VT_PART_BYTES - VT_FRAME_OFFSET - VT_GEN_BYTES - VT_FSR_BYTES - VT_VEC_BYTES - VT_CELLS_BYTES)

#define VT_MAX_LEVELS 6
#define VT_MAX_FEATURES 512

#define VT_CACHELINE __attribute__((aligned(64)))

enum vt_session_kind { VT_SESSION_NONE, VT_SESSION_XFER, VT_SESSION_TRACK };

/* What the host asks of one session: the layout, read once at its start. What can change while frames flow is soft configuration (vt_sc_key). */
struct vt_config {
    uint64_t format_hash;   /* vt_format_hash(): the card refuses a session whose layouts differ from its own */
    uint32_t kind;          /* vt_session_kind */
    uint32_t width, height; /* width a multiple of 16: rows then fall on 16 B boundaries for {uint8} */
    uint32_t stereo;        /* 0: mode A, one camera; 1: mode B, two rectified synchronized cameras */
    uint32_t slots;         /* frame slots in use */
    uint32_t cores, tpc;    /* 0 cores: every usable core */
    uint32_t pipelines;     /* 1: one team runs every stage in turn. 2: two groups of cores: pyramids alongside the rest */
    uint32_t pyr_cores;     /* with 2 pipelines: cores in the pyramid group */
    uint32_t levels;        /* pyramid levels, 1 = the image alone */
    uint32_t patch_w, patch_h;
    uint32_t max_iter;
    float eps;              /* LK stop, pixels */
    float min_eig;          /* smaller eigenvalue per patch pixel, gradients in 8-bit grey levels per pixel */
    float max_residual;     /* mean |T - J| in 8-bit grey above which a feature is dropped */
    float fb_max;           /* forward-backward miss, pixels, above which a feature is dropped; 0 = no check */
    uint32_t max_features;
    float min_dist;         /* between features, pixels */
    float corner_quality;   /* relative to the strongest in the region */
    uint32_t min_features;  /* a region with fewer usable features is not observed */
    float min_inliers;      /* fraction of features agreeing with the region's motion, for observed */
    uint32_t prefetch;      /* LK patch prefetch */
    uint32_t pre_gauss;     /* 1: a 5 x 5 binomial on the input before the pyramid */
    uint32_t contrast;      /* 1: linear stretch of the frame's min .. max to the full range */
    uint32_t frame_diff;    /* 1: |this frame - last| over the plane, reported as a mean */
    uint32_t max_missing;   /* frames unobserved before LOST */
    float reacq_ncc;        /* correlation with the region's stored appearance needed to take it back */
    uint32_t reacq_radius;  /* search radius at the appearance's level, pixels */
    /* Stereo geometry, rectified: focal length (pixels), principal point, baseline (metres). */
    float fx, fy, cx, cy, baseline;
    float min_disp, max_disp;     /* pixels */
    float min_ncc, ambiguity;     /* correspondence acceptance: the peak's correlation, second peak / peak */
    float max_reproj;             /* pixels */
    /* Measurement and process noise */
    float sigma_c;      /* the region centre, pixels */
    float sigma_d;      /* the disparity, pixels, a floor under what the features' scatter shows */
    float r_floor;      /* metres */
    float q_accel;      /* white-acceleration density: (m/s^2) per sqrt(Hz) */
    float v0_sigma;     /* initial velocity uncertainty, m/s */
    uint32_t dual;      /* 1: two cameras, each tracked on its own at its own rate and size; a calibration makes them one 3D point (fuse.h). width, height are camera A's */
    uint32_t width2, height2; /* camera B */
    uint32_t width3, height3; /* camera D, with dual: the depth map's view, no part of the triangulation. 0 = none */
    /* The depth map, with three cameras and a geometry for all of them. 0 planes = none. */
    uint32_t depth_planes, depth_agg; /* depths swept, uniform in 1 / z; aggregation radius */
    uint32_t depth_avg;      /* frames a still picture is averaged over, per camera. 1 = every map from single frames */
    float depth_near, depth_far;      /* metres */
    float depth_trunc, depth_tex, depth_conf, depth_eps; /* as struct depth_cfg */
    float motion_px;         /* a cell whose vector against its camera's mean is longer than this, in pixels of the map's level, has moved */
    uint32_t flow_level;     /* the pyramid level the still pictures and their vectors run at: 0 native, at most VT_DEPTH_LEVEL. The map reads VT_DEPTH_LEVEL regardless */
    uint32_t framegen;       /* 1: a frame is generated between real frames for the tracker of a camera with a region */
    uint32_t gen_pictures;   /* 1, with framegen: every camera's every generated frame is made whether a region exists or not, and its picture sent to the host before its result */
    float motion_grey;       /* a cell whose mean |frame - mean| is above this, in 8-bit grey, has changed: what a vector cannot see on a flat surface */
    uint32_t fsr_camera;     /* 1 + the camera whose picture goes through FSR 3 (src/kern/fsr3.h), with framegen. 0 = none */
    uint32_t fsr_w, fsr_h;   /* its size afterwards */
    uint32_t fsr_up_w, fsr_up_h; /* a first stage up to this, then a second down to fsr_w x fsr_h. 0 = one stage */
    uint32_t fsr_async;      /* 1: FSR runs on hardware threads of its own and drops frames when it is behind, and its pictures go by fsr_pub. 0: in line, and every frame: what the gates compare */
    uint32_t gen_cameras;    /* bit k: camera k gets generated pictures. 0 = all */
    float fsr_sharp;         /* RCAS, stops: 0 the sharpest */
    uint32_t vec_overlay;    /* 1: with fsr_async, the orange vectors of FSR's camera run on the core nothing else uses, and are drawn at the upscaled size before the second stage */
};

#define VT_MAX_CAMERAS 3

static inline uint32_t vt_cameras(const struct vt_config *c) { return !c->dual ? 1u : c->width3 ? 3u : 2u; }

/* How the cameras relate. X_b = R X_a + T, metres: camera A is the world frame. Radial distortion k1, k2. Sent whenever it changes, frames keep flowing. */
struct vt_calib {
    uint64_t valid;
    double fa, cxa, cya, k1a, k2a, fb, cxb, cyb, k1b, k2b;
    double R[9], T[3];
    uint64_t valid_d;   /* camera D: X_d = Rd X_a + Td */
    double fd, cxd, cyd, k1d, k2d;
    double Rd[9], Td[3];
};

enum vt_msg_type { VT_MSG_NONE, VT_MSG_FRAME, VT_MSG_RESULT };
#define VT_MSG_MAGIC 0x4d54u /* "TM" */

/* Stages a frame is for: its destination. */
#define VT_PIPE_PYR 1u
#define VT_PIPE_FLOW 2u
#define VT_PIPE_STEREO 4u
#define VT_PIPE_EST 8u
#define VT_PIPE_ALL (VT_PIPE_PYR | VT_PIPE_FLOW | VT_PIPE_STEREO | VT_PIPE_EST)

/* Priority: should the card process this before what it holds already. */
enum vt_priority { VT_PRIO_ORDERED, VT_PRIO_LIVE /* frames waiting ahead of this one are skipped */ };
/* Algorithm: which kernels this frame runs on. Per message, so both can alternate through one run. */
enum vt_algorithm { VT_ALG_MVEX, VT_ALG_X87 };
enum vt_cmd { VT_CMD_NONE, VT_CMD_SELECT, VT_CMD_DESELECT };
/* Soft configuration: changes what the pipelines do, from this frame on, with no session change and no reallocation. */
enum vt_sc_key {
    VT_SC_NONE, VT_SC_FB_MAX, VT_SC_MAX_ITER, VT_SC_EPS, VT_SC_MIN_EIG, VT_SC_MAX_RESIDUAL, VT_SC_MAX_FEATURES, VT_SC_Q_ACCEL, VT_SC_R_FLOOR, VT_SC_SIGMA_C, VT_SC_SIGMA_D,
    VT_SC_PREFETCH, VT_SC_MAX_MISSING, VT_SC_REACQ_NCC, VT_SC_CONTRAST, VT_SC_PRE_GAUSS,
    VT_SC_DEBUG_LEVEL, /* -1 off, or a pyramid level: the card sends that plane of its own memory with every result, to the transfer area. For looking, not for production. */
    VT_SC_FSR_SHARP, /* RCAS, stops: the camera going through FSR, at any moment */
    VT_SC_COUNT
};

/* A message object's header, 8-byte words: read word by word from uncached memory. */
struct vt_msg {
    uint16_t magic, type;    /* VT_MSG_MAGIC, vt_msg_type */
    uint16_t dest, priority; /* VT_PIPE_ mask, vt_priority */
    uint32_t algorithm, length; /* vt_algorithm; the bytes of data in its slot */
    uint64_t instance;       /* the source's frame number: strictly increasing */
    uint64_t acq_ns;         /* host CLOCK_MONOTONIC at acquisition, or the dataset's clock: the time every estimate is referred to */
    uint64_t pub_ns;         /* host CLOCK_MONOTONIC at publication */
    uint32_t slot, cmd;      /* where its data is; vt_cmd, applied at this frame */
    float roi[4];            /* VT_CMD_SELECT: x, y, w, h in pixels of the left image */
    uint32_t sc_key;         /* vt_sc_key */
    float sc_value;
    uint32_t camera, generated; /* which camera's tracker it is for: 0 or 1; 1 on a frame made on the card between two real ones, which never crosses the wire as a frame */
    uint64_t check;          /* vt_msg_check of everything before it */
};
_Static_assert(sizeof(struct vt_msg) == 88, "the descriptor table's stride");

static inline uint64_t vt_msg_check(const struct vt_msg *m)
{
    const uint64_t *w = (const uint64_t *)m;
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (unsigned i = 0; i < (sizeof *m - 8) / 8; i++) {
        h ^= w[i];
        h *= 0xC2B2AE3D27D4EB4Full;
        h ^= h >> 29;
    }
    return h;
}

/* What can go wrong with a message: counted, reported, the frame dropped or the bad part ignored. */
enum vt_exception { VT_EXC_NONE, VT_EXC_BAD_CHECK, VT_EXC_BAD_TYPE, VT_EXC_BAD_SLOT, VT_EXC_BAD_LENGTH, VT_EXC_INSTANCE_ORDER, VT_EXC_ROUTE_STEREO, VT_EXC_BAD_SOFTCONFIG, VT_EXC_BAD_ALGORITHM };

#define VT_DESC_RING 256

/* Two pictures from FSR made on their own time: a real instant's, and the one half way from the last. */
struct vt_fsr_meta {
    uint64_t instance;
    uint32_t slot, slot_mid, w, h;
    uint32_t pad[10];
};
_Static_assert(sizeof(struct vt_fsr_meta) == 64, "one line");

/* The orange vectors of one real frame, brought down: rows y0 to y1 of a slot are written, the others are zero and are not sent. */
struct vt_vec_meta {
    uint64_t instance;
    uint32_t slot, w, h, y0, y1, lines, threads, pad;
    uint64_t cycles_vectors, cycles_draw, cycles_down;
};
_Static_assert(sizeof(struct vt_vec_meta) == 64, "one line");

/* One frame's vectors of L or R. */
struct vt_cells_meta {
    uint64_t instance;
    uint32_t cam, slot, cols, rows, level;
    uint32_t pad[9];
};
_Static_assert(sizeof(struct vt_cells_meta) == 64, "one line");

struct vt_ctl {
    uint64_t magic;
    /* Session handshake. The host writes config, then session_seq. The card answers card_session = session_seq when it is ready,
     * or -(seq) on failure. session_end ends a session, quit ends the program. */
    volatile uint64_t session_seq;
    volatile uint64_t session_end;
    volatile uint64_t quit;
    struct vt_config config VT_CACHELINE;
    /* Host to card. */
    volatile uint64_t frames_pub VT_CACHELINE; /* messages published; written last */
    volatile uint64_t xfer_cmd;                /* the transfer benchmark's request */
    volatile uint64_t xfer_bytes;
    volatile uint64_t xfer_seq;
    /* Card to host. */
    volatile uint64_t card_session VT_CACHELINE;
    volatile uint64_t card_format;               /* the card's vt_format_hash, whatever it decided */
    volatile uint64_t frames_taken VT_CACHELINE; /* messages whose slot is free again */
    volatile uint64_t exceptions VT_CACHELINE;   /* messages dropped as malformed, this session */
    volatile uint64_t last_exception;            /* vt_exception */
    volatile uint64_t results_pub VT_CACHELINE;  /* results published: a DMA status, after the result's own copy */
    volatile uint64_t xfer_done VT_CACHELINE;
    volatile uint64_t xfer_card_cycles;
    volatile uint64_t card_build VT_CACHELINE;   /* HHMMSS of the card program's binary: which code is running */
    volatile uint64_t calib_seq VT_CACHELINE;    /* the host changes calib, then this */
    volatile uint64_t depth_pub VT_CACHELINE;    /* depth maps published: a DMA status, after the map's own copies. The latest is in buffer (depth_pub - 1) & 1 */
    struct vt_fsr_meta fsr_meta VT_CACHELINE;
    volatile uint64_t fsr_pub VT_CACHELINE;     /* pictures from FSR published, each with its meta, when it runs apart from the tracker */
    struct vt_vec_meta vec_meta VT_CACHELINE;
    volatile uint64_t vec_pub VT_CACHELINE;     /* planes of orange vectors published, each with its meta */
    struct vt_cells_meta cells_meta[VT_CELL_CAMERAS] VT_CACHELINE;
    volatile uint64_t cells_pub[VT_CELL_CAMERAS] VT_CACHELINE; /* frames of vectors published, each with its meta */
    struct vt_calib calib VT_CACHELINE;
    struct vt_msg desc[VT_DESC_RING] VT_CACHELINE;
};
_Static_assert(sizeof(struct vt_ctl) <= 65536, "the control area");

enum vt_xfer_op { VT_XFER_H2C = 1, VT_XFER_C2H = 2, VT_XFER_PING = 3 };

/* The frame size a slot takes: 1 or 2 images, each height rows of a stride of width bytes. */
static inline uint64_t vt_slot_bytes(uint32_t w, uint32_t h, uint32_t stereo)
{
    const uint64_t one = (uint64_t)w * h;
    return (((stereo ? 2 : 1) * one) + 63) & ~63ull;
}

/* A slot's size in a session: whichever camera needs most. */
static inline uint64_t vt_slot_stride(const struct vt_config *c)
{
    const uint64_t a = vt_slot_bytes(c->width, c->height, c->stereo);
    const uint64_t b = c->dual ? vt_slot_bytes(c->width2, c->height2, 0) : 0;
    const uint64_t d = c->dual && c->width3 ? vt_slot_bytes(c->width3, c->height3, 0) : 0;
    const uint64_t m = a > b ? a : b;
    return m > d ? m : d;
}

/* ---- the result of one frame ---- */

enum vt_state {
    VT_ST_NONE,         /* no region selected */
    VT_ST_NEW,          /* selected at this frame */
    VT_ST_OBSERVED,     /* measured at this frame */
    VT_ST_UNOBSERVED,   /* not measured at this frame: any position shown is a prediction */
    VT_ST_LOST,         /* given up; no position */
    VT_ST_DESELECTED,   /* the operator ended it */
};

#define VT_F_PREDICTED 1u    /* the image position is a prediction, not a measurement */
#define VT_F_REACQUIRED 2u   /* taken back at this frame from the stored appearance */
#define VT_F_NO_FEATURES 4u  /* a selection with too little texture to track */
#define VT_F_OUT_OF_VIEW 8u  /* the predicted region has left the image */
#define VT_F_METRIC 16u      /* pos/vel are valid metric estimates */
#define VT_F_3D_MEASURED 32u /* a 3D measurement was accepted at this frame */
#define VT_F_3D_REJECTED 64u /* a 3D measurement existed but failed the innovation gate */
#define VT_F_AMBIGUOUS 128u  /* correspondence too ambiguous for a measurement this frame */
#define VT_F_GENERATED 256u  /* from a frame generated between two real ones: its instance is the newer real frame's, acq_ns half way. Nothing in it is a measurement */
#define VT_F_AT_REST 512u    /* the head has not moved since the selection: the box's centre is where the operator drew it, and nothing has yet shown which part of the box is the thing */
#define VT_F_UNPROVEN 1024u  /* this 3D rests on such a box and the box holds more than one depth, or a view is at rest: its covariance says how little is known, and it is not to be read as a measurement of the thing */
#define VT_F_POINT_MID 2048u /* the box is plainly not centred on the thing, so the point is the middle of its members: not the same physical spot in another view, to within pt_sx, pt_sy */

/* Per feature. */
#define VT_FF_STEREO 1u      /* xr, yr, disparity hold a sub-pixel match */
#define VT_FF_3D 2u          /* its 3D point passed every check */
#define VT_FF_AMBIGUOUS 4u
#define VT_FF_NEW 8u         /* detected at this frame */
#define VT_FF_MEMBER 16u     /* moves with the region and votes for it; without this it is riding along, on probation */

enum vt_stage { VT_T_DMA_IN, VT_T_QUEUE, VT_T_PYR, VT_T_LK, VT_T_REGION, VT_T_STEREO, VT_T_EST, VT_T_TRACK, VT_T_DEPTH /* this frame's vectors, mean and, on camera D, map */, VT_T_GEN /* making the frame before this one */, VT_T_COUNT };

struct vt_feat_out {
    float x, y;       /* left image */
    float u, v;       /* displacement this frame */
    float xr, yr;     /* right image */
    float disparity;
    float sigma;      /* standard deviation of the position estimate, pixels: a lower bound (independent residuals) */
    uint32_t flags;
    float z;
};

struct vt_result {
    struct vt_msg hdr;  /* the same object format as the frames: type VT_MSG_RESULT, instance = the frame's, length = this result's bytes */
    uint64_t seq;       /* 1-based, the frame's place in the session */
    uint64_t last_obs_ns;  /* acquisition time of the last frame the region was measured in */
    uint32_t frames_skipped; /* frames the card discarded before this one (VT_PRIO_LIVE) */
    uint32_t state, flags, track_id;
    uint32_t missing;   /* consecutive frames unobserved */
    uint32_t nfeat, nfeat_3d;
    uint32_t exceptions; /* parts of this frame's message that were ignored: 1 << vt_exception */
    float cx, cy, w, h; /* region, left image */
    float pt_x, pt_y;   /* the point two views triangulate: the box's centre, unless its members say the box is not centred on the thing (VT_F_POINT_MID): then the middle of its members */
    float pt_sx, pt_sy; /* how well the middle of its members is known, pixels: the scatter of a median. Two views each see their own corners, so this does not average away */
    float du, dv;       /* its displacement this frame: observed frames only */
    float scale;        /* its image scale change this frame */
    float rate_u, rate_v; /* pixels per second: image-plane, never metric */
    float inliers;      /* fraction of features agreeing */
    float quality;      /* 0 .. 1 */
    float sigma_px;     /* region position standard deviation, pixels: a lower bound */
    float motion;       /* mean |frame - last frame| in 8-bit grey, when asked for */
    float reacq_ncc;    /* unobserved frames: the best correlation with the stored appearance, and where, in pixels of the image */
    float reacq_x, reacq_y;
    uint32_t picture;   /* generated frames: 1 + the slot of this camera's picture (vt_gen_offset); 0 none. It has landed by the time this result is published */
    uint32_t fsr_slot;  /* 1 + the slot of this instant's picture from FSR (vt_fsr_offset); 0 none. Landed when this result says so */
    uint32_t fsr_w, fsr_h;
    int32_t debug_level; /* the plane sent with this result, -1 none: its size, and its stride in lanes */
    uint32_t debug_w, debug_h, debug_stride;
    float pair_ua, pair_va; /* two views: camera A's centre as carried to camera B's time */
    double parallax_deg, pair_gap_ms; /* the angle between the two rays; how far camera A's frame was carried */
    double meas[3], meas_cov[9];  /* this frame's 3D measurement, metres (VT_F_3D_MEASURED or _REJECTED) */
    double nis;         /* its normalized innovation */
    double reproj_rms;  /* median over its features, pixels */
    double pos[3], vel[3];   /* metres, m/s (VT_F_METRIC) */
    double pos_cov[9];  /* P's position block */
    double vel_var[3];  /* P's velocity diagonal */
    double age_s;       /* seconds since the last accepted 3D measurement: a prediction's age */
    uint64_t cycles[VT_T_COUNT]; /* card TSC, per stage */
    struct vt_feat_out feat[VT_MAX_FEATURES];
};
_Static_assert(sizeof(struct vt_result) <= VT_RES_BYTES, "a result slot");

/* ---- the depth map, and the vectors behind it ---- */

#define VT_DEPTH_LEVEL 2      /* the pyramid level of each camera that the map and the vectors read */
#define VT_CELL_W 16          /* a camera's picture in cells, in pixels of that level: one vector's lanes across */
#define VT_CELL_H 8

/* state of a cell */
#define VT_CELL_LK 0xffu      /* the low byte: LK's status, 0 = a good vector */
#define VT_CELL_CHANGED 0x100u /* this cell's own evidence said it moved */
#define VT_CELL_MOVING 0x200u /* it, or a neighbour, moved: its mean starts again */
#define VT_CELL_COUNT_SHIFT 16 /* the high half: frames in its mean */

/* One cell of one camera at one frame: LK from that camera's long-term mean to this frame, at the cell's centre. */
struct vt_cell {
    float u, v;     /* pixels of the map's level; meaningful when LK's status is 0 */
    float diff;     /* mean |frame - mean| over the cell, 8-bit grey */
    uint32_t state;
    float sigma;    /* the vector's standard deviation from LK's own covariance, pixels of the level: a lower bound */
    float pad;
};

struct vt_motion {
    uint32_t cols, rows;
    uint32_t cell_at;   /* bytes from the buffer's start, vt_cell x cols x rows */
    uint32_t tracked;   /* cells with a good vector */
    uint32_t moving;    /* cells that moved at the newest frame */
    float count_mean;   /* frames in a cell's mean */
    float drift_u, drift_v; /* the mean vector over cells that did not move, pixels of the map's level: the camera itself against its own past */
    float drift_rms;    /* those vectors' scatter about it */
    float diff_median;  /* over all cells: the camera's noise, 8-bit grey */
    float diff_limit;   /* what a cell's difference had to exceed: motion_grey, or this camera's noise times DIFF_OVER_NOISE */
    float pad;
    uint64_t cycles_vectors, cycles_mean; /* card TSC, the newest frame */
    uint64_t frames;    /* of this camera, this session */
    /* How far this camera has turned since the session's first picture of it: attitude.h. */
    double att_q[4];    /* w x y z: a bearing b in that pose is seen at q b q' now */
    float att_corr, att_corr_none; /* the picture against its reference, turned and not turned */
    float att_used, att_view;      /* the kernel's share of the pixels in view, and the share in view */
    float att_peak_z;   /* the search's peak, in standard deviations */
    uint32_t att_state; /* VT_ATT_* */
    uint32_t att_keys;  /* references taken this session */
    uint32_t att_pad;
    uint64_t cycles_att; /* card TSC, the newest measurement */
};
enum { VT_ATT_NONE, VT_ATT_HELD, VT_ATT_TURNED };

/* What each buffer starts with. Then z, F32 metres, 0 = unknown, h rows of stride lanes at VT_DEPTH_Z_AT; conf, the same size, after it; then each camera's cells. */
struct vt_depth_out {
    uint64_t seq;        /* 1-based */
    uint64_t instance;   /* camera D's frame */
    uint64_t acq_ns;     /* its acquisition time */
    uint64_t geometry;   /* the calib_seq it was computed with */
    uint32_t w, h, stride, planes;
    float z_near, z_far;
    float f, cx, cy;     /* the reference's, at this level: what turns a pixel and a depth into a point */
    uint32_t flow_level; /* of the cells below */
    uint32_t textured;   /* pixels with enough texture for a depth */
    uint32_t confident;  /* of those, confidence at least 0.2: the number every offline figure quotes */
    uint32_t known;      /* pixels with a depth */
    uint32_t views;
    uint64_t cycles_normalize, cycles_sweep, cycles_aggregate, cycles_decide;
    struct vt_motion cam[VT_MAX_CAMERAS];
};
#define VT_DEPTH_Z_AT 4096u
_Static_assert(sizeof(struct vt_depth_out) <= VT_DEPTH_Z_AT, "a depth buffer's head");

static inline const float *vt_depth_z(const void *buf) { return (const float *)(const void *)((const char *)buf + VT_DEPTH_Z_AT); }
static inline const float *vt_depth_conf(const void *buf)
{
    const struct vt_depth_out *h = (const struct vt_depth_out *)buf;
    return vt_depth_z(buf) + (uint64_t)h->stride * h->h;
}
static inline const struct vt_cell *vt_depth_cells(const void *buf, int cam)
{
    const struct vt_depth_out *h = (const struct vt_depth_out *)buf;
    return (const struct vt_cell *)(const void *)((const char *)buf + h->cam[cam].cell_at);
}

#endif
