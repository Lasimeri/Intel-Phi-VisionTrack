/* run_opts.c: see run_opts.h. */
#include "run_opts.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const sc_names[VT_SC_COUNT] = {"none", "fb_max", "max_iter", "eps", "min_eig", "max_residual", "max_features", "q_accel", "r_floor", "sigma_c", "sigma_d", "prefetch", "max_missing", "reacq_ncc", "contrast", "pre_gauss", "debug_level", "fsr_sharp"};

int run_opts_parse(struct run_opts *o, int argc, char **argv, struct dataset *ds)
{
    memset(o, 0, sizeof *o);
    o->bad_at = -1;
    o->bad_route = -1;
    o->deselect_at = -1;
    o->nframes = 150;
    o->fps = 30.0;
    if (argc < 3) {
        fprintf(stderr, "usage: %s SCENE WxH [--frames N --fps F --mono --cores C --tpc T --pipes 1|2 --pyr-cores N --variant V --alternate --paced --drop --live --levels L --patch W H --feat N\n"
                        "    --contrast --pregauss --framediff --slots S --noselect --deselect-at F --q Q --sigc S --sigd S --rfloor R --sc FRAME:KEY:VALUE --bad-at F --bad-kind K --csv F --json F --hash F --gate]\n", argv[0]);
        return -1;
    }
    o->id = scene_by_name(argv[1]);
    if (o->id == SC_COUNT || sscanf(argv[2], "%dx%d", &o->w, &o->h) != 2) {
        fprintf(stderr, "%s: bad scene or size\n", argv[0]);
        return -1;
    }
    /* A first pass for what the dataset needs. */
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) o->nframes = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--fps") && i + 1 < argc) o->fps = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--mono")) o->mono = 1;
    }
    if (dataset_open(ds, o->id, o->w, o->h, o->nframes, o->fps))
        return -1;
    struct vt_config *c = &o->cfg;
    vt_default_config(c, &ds->s, !o->mono);
    for (int i = 3; i < argc; i++) {
#define OPT(name) (!strcmp(argv[i], name) && i + 1 < argc)
        if (OPT("--frames") || OPT("--fps")) i++;
        else if (!strcmp(argv[i], "--mono")) continue;
        else if (!strcmp(argv[i], "--dual")) {
            o->dual = 1;
            c->dual = 1;
            c->stereo = 0;
            c->width2 = (uint32_t)o->w;
            c->height2 = (uint32_t)o->h;
        }
        else if (!strcmp(argv[i], "--framegen")) c->framegen = c->gen_pictures = 1;
        else if (!strcmp(argv[i], "--fsr-up") && i + 1 < argc) {
            int fw, fh;
            if (sscanf(argv[++i], "%dx%d", &fw, &fh) != 2)
                return 2;
            c->fsr_up_w = (uint32_t)fw;
            c->fsr_up_h = (uint32_t)fh;
        }
        else if (!strcmp(argv[i], "--fsr-sharp") && i + 1 < argc) c->fsr_sharp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--fsr") && i + 1 < argc) {
            int fw, fh;
            if (sscanf(argv[++i], "%dx%d", &fw, &fh) != 2)
                return 2;
            c->framegen = c->gen_pictures = 1;
            c->fsr_camera = 1;
            c->fsr_w = (uint32_t)fw;
            c->fsr_h = (uint32_t)fh;
        }
        else if (OPT("--cores")) c->cores = (uint32_t)atoi(argv[++i]);
        else if (OPT("--tpc")) c->tpc = (uint32_t)atoi(argv[++i]);
        else if (OPT("--pipes")) c->pipelines = (uint32_t)atoi(argv[++i]);
        else if (OPT("--pyr-cores")) c->pyr_cores = (uint32_t)atoi(argv[++i]);
        else if (OPT("--variant")) o->variant = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--alternate")) o->alternate = 1;
        else if (!strcmp(argv[i], "--paced")) o->paced = 1;
        else if (!strcmp(argv[i], "--drop")) o->drop = 1;
        else if (!strcmp(argv[i], "--live")) o->live = 1;
        else if (!strcmp(argv[i], "--noselect")) o->noselect = 1;
        else if (OPT("--deselect-at")) o->deselect_at = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gate")) o->gate = 1;
        else if (!strcmp(argv[i], "--contrast")) c->contrast = 1;
        else if (!strcmp(argv[i], "--pregauss")) c->pre_gauss = 1;
        else if (!strcmp(argv[i], "--framediff")) c->frame_diff = 1;
        else if (OPT("--csv")) o->csv = argv[++i];
        else if (OPT("--json")) o->json = argv[++i];
        else if (OPT("--hash")) o->hash = argv[++i];
        else if (OPT("--levels")) c->levels = (uint32_t)atoi(argv[++i]);
        else if (OPT("--slots")) c->slots = (uint32_t)atoi(argv[++i]);
        else if (OPT("--feat")) c->max_features = (uint32_t)atoi(argv[++i]);
        else if (OPT("--q")) c->q_accel = (float)atof(argv[++i]);
        else if (OPT("--sigc")) c->sigma_c = (float)atof(argv[++i]);
        else if (OPT("--sigd")) c->sigma_d = (float)atof(argv[++i]);
        else if (OPT("--rfloor")) c->r_floor = (float)atof(argv[++i]);
        else if (OPT("--bad-at")) o->bad_at = atoi(argv[++i]);
        else if (OPT("--bad-kind")) o->bad_kind = atoi(argv[++i]);
        else if (OPT("--bad-route")) o->bad_route = atoi(argv[++i]);
        else if (OPT("--sc")) {
            char key[32];
            float v;
            int f;
            if (sscanf(argv[++i], "%d:%31[^:]:%f", &f, key, &v) != 3 || o->nsc == RUN_MAX_SC) {
                fprintf(stderr, "%s: bad --sc %s\n", argv[0], argv[i]);
                return -1;
            }
            uint32_t k = VT_SC_COUNT + 1; /* a key the card must refuse, on purpose, if asked for by a name it lacks */
            for (uint32_t s = 0; s < VT_SC_COUNT; s++)
                if (!strcmp(key, sc_names[s]))
                    k = s;
            o->sc[o->nsc++] = (struct run_sc){f, k, v};
        } else if (!strcmp(argv[i], "--patch") && i + 2 < argc) {
            c->patch_w = (uint32_t)atoi(argv[++i]);
            c->patch_h = (uint32_t)atoi(argv[++i]);
        } else {
            fprintf(stderr, "%s: unknown option %s\n", argv[0], argv[i]);
            return -1;
        }
    }
    return 0;
}

void run_opts_msg(const struct run_opts *o, const struct dataset *ds, int i, int camera, struct vt_msg *m)
{
    memset(m, 0, sizeof *m);
    m->magic = VT_MSG_MAGIC;
    m->type = VT_MSG_FRAME;
    m->camera = (uint32_t)camera;
    m->dest = (uint16_t)(o->cfg.stereo || i == o->bad_route ? VT_PIPE_ALL : (VT_PIPE_PYR | VT_PIPE_FLOW | VT_PIPE_EST));
    m->priority = o->live ? VT_PRIO_LIVE : VT_PRIO_ORDERED;
    m->algorithm = o->alternate ? (uint32_t)(i & 1) : (uint32_t)o->variant;
    m->length = (uint32_t)vt_slot_bytes(o->cfg.width, o->cfg.height, o->cfg.stereo);
    m->instance = (uint64_t)i;
    m->acq_ns = o->dual ? dataset_acq_ns_cam(ds, i, camera) : dataset_acq_ns(ds, i);
    if (i == ds->s.select_frame && !o->noselect) {
        m->cmd = VT_CMD_SELECT;
        dataset_roi_cam(ds, i, camera, m->roi);
    } else if (i == o->deselect_at) {
        m->cmd = VT_CMD_DESELECT;
    }
    for (int k = 0; k < o->nsc; k++)
        if (o->sc[k].frame == i) {
            m->sc_key = o->sc[k].key;
            m->sc_value = o->sc[k].value;
        }
}
