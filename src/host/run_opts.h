/* run_opts.h: one scene run's options, parsed once, and the message stream they imply. vt_run and vt_replay both use it, so a replay receives exactly the messages the card did. */
#ifndef VT_RUN_OPTS_H
#define VT_RUN_OPTS_H

#include "dataset.h"
#include "vt_proto.h"

#define RUN_MAX_SC 32

struct run_sc { int frame; uint32_t key; float value; };

struct run_opts {
    enum scene_id id;
    int w, h, nframes;
    double fps;
    int mono, dual, noselect, gate, deselect_at;
    int paced, drop, live;
    int alternate;      /* frames alternate MVEX and x87 */
    int variant;        /* every frame this algorithm */
    int bad_at;         /* publish a malformed message before this frame, -1 none */
    int bad_kind;       /* which malformation: vt_exception */
    int bad_route;      /* this frame asks for a stage the session does not have, -1 none */
    const char *csv, *json, *hash;
    int nsc;
    struct run_sc sc[RUN_MAX_SC];
    struct vt_config cfg;
};

/* Parses "SCENE WxH [options]"; 0, or -1 with a message. */
int run_opts_parse(struct run_opts *o, int argc, char **argv, struct dataset *ds);
/* The header of frame i, without slot, pub_ns, check. */
void run_opts_msg(const struct run_opts *o, const struct dataset *ds, int i, int camera, struct vt_msg *m);

#endif
