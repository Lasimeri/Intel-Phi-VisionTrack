/* score.h: a run's results against the scene's analytic truth. The same code scores the card's run and the host's replay. */
#ifndef VT_SCORE_H
#define VT_SCORE_H

#include <stdio.h>

#include "dataset.h"
#include "vt_proto.h"

struct score;
struct score *score_new(const struct dataset *ds, const char *csv_path);
/* One result, its latency in microseconds on the host's clock (0 in a replay). */
void score_frame(struct score *s, const struct vt_result *r, double lat_us, double us_per_cycle);
/* The summary as one JSON object. */
void score_print(struct score *s, FILE *f, const char *mode, int results, int dropped, double wall_s, const char *extra);
/* The scene's documented tolerances (tests/scenes.md). Prints each one it misses; returns how many. */
int score_gate(struct score *s, FILE *f);

#endif
