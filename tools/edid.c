/* edid.c: each connected output's native mode, physical size and name, from the EDID's first detailed timing and descriptors. */
#include <stdio.h>
int main(int argc, char **argv)
{
    for (int a = 1; a < argc; a++) {
        unsigned char e[256];
        FILE *f = fopen(argv[a], "rb");
        if (!f)
            continue;
        const size_t n = fread(e, 1, sizeof e, f);
        fclose(f);
        if (n < 128)
            continue;
        const unsigned char *d = e + 54;
        const int hp = d[2] | (d[4] & 0xf0) << 4, vp = d[5] | (d[7] & 0xf0) << 4;
        const int hm = d[12] | (d[14] & 0xf0) << 4, vm = d[13] | (d[14] & 0x0f) << 8;
        char name[14] = "";
        for (int k = 54; k < 126; k += 18)
            if (!e[k] && !e[k + 1] && e[k + 3] == 0xfc) {
                int i = 0;
                for (; i < 13 && e[k + 5 + i] != 10; i++)
                    name[i] = (char)e[k + 5 + i];
                name[i] = 0;
            }
        printf("%-44s %5d x %-5d px  %4d x %-4d mm  coarse %3d x %-3d cm  %s\n", argv[a], hp, vp, hm, vm, e[21], e[22], name);
    }
    return 0;
}
