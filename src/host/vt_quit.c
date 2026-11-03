/* vt_quit.c: tells vtrackd on the card to leave, through the region: no control-socket call. */
#include <stdio.h>
#include <time.h>

#include "vt_link.h"

int main(void)
{
    struct vt_link l;
    if (vt_link_open(&l))
        return 2;
    l.ctl->session_end = 1;
    l.ctl->quit = 1;
    vt_link_close(&l);
    printf("host: vtrackd asked to quit\n");
    return 0;
}
