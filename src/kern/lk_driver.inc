/* lk_driver.inc: the pyramid walk, included once per arithmetic. The includer defines three operations and LK_NAME:
 *   build:    samples the template at (px, py), fills T, Tx, Ty in the tile, returns the structure tensor's sums
 *   iterate:  samples J at (qx, qy), returns the sums of It Tx and It Ty, in Q6 grey x 8-bit grey
 *   residual: the same sampling, returns the mean |It| and the mean It^2, 8-bit grey
 * Everything here runs in long double: the same 64-bit-mantissa arithmetic on both machines. */

void LK_NAME(const struct vt_plane *I, const struct vt_plane *J, int top, const struct lk_cfg *c, float x, float y, float gx0, float gy0, struct lk_out *o, void *tile)
{
    long double gx = (long double)gx0 * lk_pow2m[top], gy = (long double)gy0 * lk_pow2m[top];
    const long double area = (long double)(c->w * c->h);
    const long double eps2 = (long double)c->eps * (long double)c->eps;
    long double det0 = 0, gxx0 = 0, gxy0 = 0, gyy0 = 0, qx = x, qy = y;
    int status = 0, iters = 0, valid0 = 0;
    float eig0 = 0.0f;
    for (int L = top; L >= 0; L--) {
        const long double s = lk_pow2m[L];
        const long double px = (long double)x * s, py = (long double)y * s;
        const int mx = c->w / 2 + 1, my = c->h / 2 + 1;
        const long double last_x = (long double)(I[L].w - mx - 1), last_y = (long double)(I[L].h - my - 1);
        if (px < mx || py < my || px >= last_x || py >= last_y) {
            /* the template does not fit at this level: at level 0 that is a lost point, above it the level is skipped */
            if (L == 0)
                status |= LK_OUT;
            else {
                gx *= 2;
                gy *= 2;
            }
            continue;
        }
        float g[3];
        LK_BUILD(&I[L], px, py, c, tile, g);
        const long double gxx = g[0], gxy = g[1], gyy = g[2];
        const long double det = gxx * gyy - gxy * gxy;
        const long double tr = gxx + gyy, diff = gxx - gyy;
        const long double eig = (tr - sqrtl(diff * diff + 4 * gxy * gxy)) * 0.5L / area;
        if (L == 0)
            eig0 = (float)eig;
        if (eig < (long double)c->min_eig || det <= 0) {
            if (L == 0)
                status |= LK_LOWEIG;
            else {
                gx *= 2;
                gy *= 2;
            }
            continue;
        }
        if (L == 0) {
            det0 = det;
            gxx0 = gxx;
            gxy0 = gxy;
            gyy0 = gyy;
            valid0 = 1;
        }
        /* G does not change inside a level: one inverse here, no fdiv in the iterations. */
        const long double ia = gyy / det, ib = -gxy / det, ic = gxx / det;
        long double vx = 0, vy = 0;
        for (int it = 0; it < c->max_iter; it++) {
            const long double qxl = px + gx + vx, qyl = py + gy + vy;
            const long double lx = (long double)(J[L].w - mx - 1), ly = (long double)(J[L].h - my - 1);
            if (qxl < mx || qyl < my || qxl >= lx || qyl >= ly) {
                if (L == 0) {
                    status |= LK_OUT;
                    valid0 = 0;
                }
                break;
            }
            float b[2];
            LK_ITERATE(&J[L], qxl, qyl, c, tile, b);
            const long double bx = (long double)b[0] / VT_Q6, by = (long double)b[1] / VT_Q6;
            const long double ex = ia * bx + ib * by, ey = ib * bx + ic * by;
            vx += ex;
            vy += ey;
            iters++;
            if (ex * ex + ey * ey < eps2)
                break;
        }
        gx += vx;
        gy += vy;
        if (L > 0) {
            gx *= 2;
            gy *= 2;
        }
    }
    qx = (long double)x + gx;
    qy = (long double)y + gy;
    o->x = (float)qx;
    o->y = (float)qy;
    o->min_eig = eig0;
    o->iters = (uint16_t)iters;
    o->residual = 0.0f;
    o->cov_xx = o->cov_xy = o->cov_yy = 0.0f;
    if (valid0 && !status) {
        const int mx = c->w / 2 + 1, my = c->h / 2 + 1;
        if (qx < mx || qy < my || qx >= (long double)(J[0].w - mx - 1) || qy >= (long double)(J[0].h - my - 1)) {
            status |= LK_OUT;
        } else {
            float r[2];
            LK_RESIDUALFN(&J[0], qx, qy, c, tile, r);
            o->residual = r[0];
            if (r[0] > c->max_residual)
                status |= LK_RESIDUAL;
            const long double s2 = (long double)r[1] * area / (area - 2);
            o->cov_xx = (float)(s2 * gyy0 / det0);
            o->cov_xy = (float)(-s2 * gxy0 / det0);
            o->cov_yy = (float)(s2 * gxx0 / det0);
        }
    }
    o->status = (uint16_t)status;
}

void LK_FLOWBACK_NAME(const struct vt_plane *I, const struct vt_plane *J, int top, const struct lk_cfg *c, float x, float y, float max_fb, struct lk_out *o, void *tile)
{
    if (o->status)
        return;
    struct lk_out b;
    /* the same tile, so the template is rebuilt from J: tracking is not symmetric, that is the point of the check */
    LK_NAME(J, I, top, c, o->x, o->y, (float)((long double)x - (long double)o->x), (float)((long double)y - (long double)o->y), &b, tile);
    const long double ex = (long double)b.x - (long double)x, ey = (long double)b.y - (long double)y;
    if (b.status || ex * ex + ey * ey > (long double)max_fb * (long double)max_fb)
        o->status |= LK_FLOWBACK;
}
