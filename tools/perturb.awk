# Rotates camera D's Rd by a given angle about y, Td with it: a deliberately wrong pose, the control.
BEGIN { th = ang * 3.14159265358979 / 180; c = cos(th); s = sin(th); n = 0 }
/^#/ { print; next }
{ n++; if (n <= 7 || n == 8 && 0) { print; next }
  if (n >= 8 && n <= 10) { R[n - 7, 1] = $1; R[n - 7, 2] = $2; R[n - 7, 3] = $3; if (n == 10) { for (k = 1; k <= 3; k++) { a[k] = c * R[1, k] + s * R[3, k]; b[k] = R[2, k]; d[k] = -s * R[1, k] + c * R[3, k] } printf "%.12g %.12g %.12g\n%.12g %.12g %.12g\n%.12g %.12g %.12g\n", a[1], a[2], a[3], b[1], b[2], b[3], d[1], d[2], d[3] } next }
  if (n == 11) { printf "%.9g %.9g %.9g\n", c * $1 + s * $3, $2, -s * $1 + c * $3; next }
  print }
