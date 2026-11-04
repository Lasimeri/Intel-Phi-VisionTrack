/* vt_format.h: the layouts of everything on the wire, described as data. Host and card each hash this table; a session starts only when the two hashes agree, so a stale
 * program on the other side is refused with a reason instead of misreading bytes. After the message-format registry of US7418574B2 (docs/architecture.md). */
#ifndef VT_FORMAT_H
#define VT_FORMAT_H

#include <stddef.h>
#include <stdint.h>

#include "vt_proto.h"

struct vt_field {
    const char *name;
    uint32_t offset, size;
};

#define VT_F(T, f) {#T "." #f, (uint32_t)offsetof(T, f), (uint32_t)sizeof(((T *)0)->f)}
typedef struct vt_config vt_cfg_t;
typedef struct vt_msg vt_msg_t;
typedef struct vt_result vt_res_t;
typedef struct vt_feat_out vt_feat_t;
typedef struct vt_ctl vt_ctl_t;

static const struct vt_field vt_fields[] = {
    {"sizeof.config", 0, (uint32_t)sizeof(struct vt_config)},
    {"sizeof.msg", 0, (uint32_t)sizeof(struct vt_msg)},
    {"sizeof.result", 0, (uint32_t)sizeof(struct vt_result)},
    {"sizeof.feat", 0, (uint32_t)sizeof(struct vt_feat_out)},
    {"sizeof.ctl", 0, (uint32_t)sizeof(struct vt_ctl)},
    VT_F(vt_cfg_t, kind), VT_F(vt_cfg_t, width), VT_F(vt_cfg_t, stereo), VT_F(vt_cfg_t, pipelines), VT_F(vt_cfg_t, levels), VT_F(vt_cfg_t, eps), VT_F(vt_cfg_t, fb_max),
    VT_F(vt_cfg_t, max_features), VT_F(vt_cfg_t, prefetch), VT_F(vt_cfg_t, reacq_ncc), VT_F(vt_cfg_t, fx), VT_F(vt_cfg_t, baseline), VT_F(vt_cfg_t, sigma_c), VT_F(vt_cfg_t, q_accel),
    VT_F(vt_msg_t, type), VT_F(vt_msg_t, dest), VT_F(vt_msg_t, algorithm), VT_F(vt_msg_t, instance), VT_F(vt_msg_t, acq_ns), VT_F(vt_msg_t, slot), VT_F(vt_msg_t, roi), VT_F(vt_msg_t, sc_key), VT_F(vt_msg_t, check),
    VT_F(vt_res_t, hdr), VT_F(vt_res_t, seq), VT_F(vt_res_t, state), VT_F(vt_res_t, nfeat), VT_F(vt_res_t, cx), VT_F(vt_res_t, meas), VT_F(vt_res_t, pos), VT_F(vt_res_t, pos_cov), VT_F(vt_res_t, cycles), VT_F(vt_res_t, feat),
    VT_F(vt_feat_t, x), VT_F(vt_feat_t, disparity), VT_F(vt_feat_t, z),
    VT_F(vt_cfg_t, dual), VT_F(vt_cfg_t, width2), VT_F(vt_cfg_t, width3), VT_F(vt_msg_t, camera), VT_F(vt_ctl_t, calib_seq), VT_F(vt_ctl_t, calib),
    {"sizeof.depth", 0, (uint32_t)sizeof(struct vt_depth_out)}, {"sizeof.cell", 0, (uint32_t)sizeof(struct vt_cell)},
    VT_F(vt_cfg_t, framegen), VT_F(vt_cfg_t, gen_pictures), VT_F(vt_res_t, picture), VT_F(vt_cfg_t, vec_overlay), VT_F(vt_cfg_t, fsr_camera), VT_F(vt_cfg_t, fsr_up_w), VT_F(vt_ctl_t, fsr_pub), VT_F(vt_cfg_t, fsr_async), VT_F(vt_res_t, fsr_slot), VT_F(vt_res_t, pt_x), VT_F(vt_msg_t, generated), VT_F(vt_cfg_t, depth_planes), VT_F(vt_cfg_t, depth_avg), VT_F(vt_cfg_t, motion_grey), VT_F(vt_cfg_t, flow_level), VT_F(vt_ctl_t, depth_pub),
    VT_F(vt_ctl_t, config), VT_F(vt_ctl_t, frames_pub), VT_F(vt_ctl_t, card_session), VT_F(vt_ctl_t, frames_taken), VT_F(vt_ctl_t, results_pub), VT_F(vt_ctl_t, desc),
};

static inline uint64_t vt_format_hash(void)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < sizeof vt_fields / sizeof vt_fields[0]; i++) {
        for (const char *p = vt_fields[i].name; *p; p++) {
            h ^= (uint8_t)*p;
            h *= 0x100000001b3ull;
        }
        const uint64_t v = (uint64_t)vt_fields[i].offset << 32 | vt_fields[i].size;
        for (int b = 0; b < 64; b += 8) {
            h ^= (v >> b) & 255;
            h *= 0x100000001b3ull;
        }
    }
    return h;
}

#endif
