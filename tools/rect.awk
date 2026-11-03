# rect.awk: reads vt_calib --report. Posters are rectangles: their corner angles in 3D must be 90 degrees, whatever their size.
function ang(a, b, c,    u1, u2, u3, v1, v2, v3, d, nu, nv) {
    u1 = X[a] - X[b]; u2 = Y[a] - Y[b]; u3 = Z[a] - Z[b];
    v1 = X[c] - X[b]; v2 = Y[c] - Y[b]; v3 = Z[c] - Z[b];
    d = u1 * v1 + u2 * v2 + u3 * v3;
    nu = sqrt(u1 * u1 + u2 * u2 + u3 * u3); nv = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
    if (nu == 0 || nv == 0) return 0;
    return atan2(sqrt(1 - (d / (nu * nv)) ^ 2), d / (nu * nv)) * 57.29578;
}
function len(a, b) { return sqrt((X[a] - X[b]) ^ 2 + (Y[a] - Y[b]) ^ 2 + (Z[a] - Z[b]) ^ 2); }
/^fa / { fa = $2; fb = $9 }
/^rotation/ { rot = $5 }
/^scale/ { base = $(NF - 1) }
/^(poster|tv-|led)/ { X[$1] = $4; Y[$1] = $5; Z[$1] = $6 }
END {
    a1 = ang("posterL-TR", "posterL-TL", "posterL-BL");
    a2 = ang("posterL-TL", "posterL-TR", "posterL-BR");
    a3 = ang("posterL-TL", "posterL-BL", "posterL-BR");
    a4 = ang("posterL-TR", "posterL-BR", "posterL-BL");
    a5 = ang("posterR-TL", "posterR-BL", "posterR-BR");
    bad = (a1 - 90) ^ 2 + (a2 - 90) ^ 2 + (a3 - 90) ^ 2 + (a4 - 90) ^ 2 + (a5 - 90) ^ 2;
    printf "fb %6.1f fa %6.1f rot %5.1f base %5.2f | corners %5.1f %5.1f %5.1f %5.1f | R-poster %5.1f | rms off 90: %5.1f | posterL %.2fx%.2f m, posterR-bottom %.2f m | TV Z %.2f\n", fb, fa, rot, base, a1, a2, a3, a4, a5, sqrt(bad / 5), len("posterL-TL", "posterL-TR"), len("posterL-TL", "posterL-BL"), len("posterR-BL", "posterR-BR"), Z["tv-screen-TR"];
}
