/* autoref.h: where a camera holds still, from its own vectors, for the static regions L and R keep as camera D's reference (the operator's order, 2026-11-03).
 *
 * Per cell: how many of its frames it has gone unmoved, and how many of those camera D was moving: still while D moved is what makes a place a reference for D. Any window of cells
 * that has not moved lately and has enough corners to hold is a candidate, and the one that has stood through the most of D's motion wins. Motion and texture only: nothing here knows what anything is. */
#ifndef VT_AUTOREF_H
#define VT_AUTOREF_H

#include "vt_proto.h"

struct autoref;

struct autoref *autoref_create(int cols, int rows);
void autoref_destroy(struct autoref *a);
/* One frame's cells; d_moving: camera D is moving in it. */
void autoref_feed(struct autoref *a, const struct vt_cell *cells, int d_moving);
/* The best window of wc x wr cells: 1 and its first column and row, or 0 if none has held still with enough corners. */
int autoref_best(struct autoref *a, int wc, int wr, int *col, int *row);
/* How many cells of that window moved in the last frame fed. */
int autoref_moved(const struct autoref *a, int col, int row, int wc, int wr);

#endif
