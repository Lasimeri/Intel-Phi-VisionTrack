# Makefile: builds both halves. PHI_TREE has spaces and is passed to scripts only, never a prerequisite.
#   make            host + card programs, every card binary ISA-audited
#   make host       host tools and the host lane model of every kernel
#   make card       card programs (KNC clang -c, knc-cc link, phi-isa-audit must read 0 illegal, 0 suspect)
#   make check      host tests, then the gates on the card (exit 77 when the card is down)
#   make compdb     compile_commands.json + clangd, 0 diagnostics or exit 1
#   make clean      removes the built programs (rendered datasets stay in out/data)
include config.mk
MAKEFLAGS += -j$(JOBS) -Oline --no-builtin-rules

COMMON_INC := -Isrc/common -Isrc/kern -Isrc/card -Isrc/host
WARN := -Wall -Wextra -Werror
# -ffp-contract=off: a*b+c stays two roundings unless the source asks for a fused operation.
# -fno-math-errno: lrintl and sqrtl become fistp and fsqrt in line, with no libm call.
HOST_CFLAGS := -std=gnu11 -O2 $(WARN) -ffp-contract=off -fno-math-errno $(COMMON_INC) -MMD -MP
CARD_CFLAGS := --target=x86_64-unknown-linux-musl -march=knc --sysroot=$(KNC_SYSROOT) -std=gnu11 -O2 $(WARN) \
               -ffp-contract=off -fno-math-errno -fno-jump-tables -falign-loops=1 -fPIE $(COMMON_INC) -MMD -MP

CARD_PROGS := vt_m0 vt_m1 vt_m2 vt_m3 vtrackd
HOST_PROGS := m0_ref m1_ref m2_ref m3_ref vt_xfer vt_quit vt_run vt_replay vt_view vt_calib vt_stitch vt_align vt_gridpng vt_rigcheck vt_stars vt_mapstat vt_mapcheck vt_attitude vt_agree vt_ncc_check

KERN_ALL := kern/img_ref kern/img_vx kern/lk_ref kern/lk_vx kern/detect kern/framegen kern/stereo kern/kalman kern/fuse kern/track kern/depth kern/still kern/attitude
HOST_LIB := host/vt_link host/dataset host/synth host/run_opts host/score

vt_m0_OBJS := card/m0_gate kern/m0_kern card/topo card/team
m0_ref_OBJS := host/m0_ref kern/m0_kern
vt_m1_OBJS := card/m1_gate kern/img_ref kern/img_vx card/arena card/topo card/team
m1_ref_OBJS := host/m1_ref kern/img_ref kern/img_vx
vt_m2_OBJS := card/m2_gate kern/lk_ref kern/lk_vx kern/img_ref kern/img_vx card/arena card/topo card/team
m2_ref_OBJS := host/m2_ref kern/lk_ref kern/lk_vx kern/img_ref kern/img_vx
vt_m3_OBJS := card/m3_gate kern/depth kern/attitude kern/vx_gather kern/still kern/lk_ref kern/lk_vx kern/img_ref kern/img_vx card/arena card/topo card/team
m3_ref_OBJS := host/m3_ref host/calib host/stars kern/depth kern/attitude kern/still kern/lk_ref kern/lk_vx kern/img_ref kern/img_vx
vtrackd_OBJS := card/vtrackd card/session_xfer card/session_track card/live_depth card/vt_dma card/arena card/topo card/team kern/vx_gather $(KERN_ALL)
vt_xfer_OBJS := host/vt_xfer host/vt_link
vt_quit_OBJS := host/vt_quit host/vt_link
vt_run_OBJS := host/vt_run $(HOST_LIB)
vt_replay_OBJS := host/vt_replay $(HOST_LIB) $(KERN_ALL)
vt_ncc_check_OBJS := host/vt_ncc_check $(KERN_ALL)
vt_view_OBJS := host/vt_view host/depthview host/v4l2cap host/x11view host/vt_link host/dataset host/synth host/calib host/stars kern/attitude kern/depth kern/img_ref kern/img_vx
vt_view_LIBS := -lX11 -lXext -lturbojpeg
vt_calib_OBJS := host/vt_calib host/calib
vt_stitch_OBJS := host/vt_stitch host/calib
vt_align_OBJS := host/vt_align host/calib host/stars kern/depth kern/img_ref kern/img_vx
vt_gridpng_OBJS := host/vt_gridpng
vt_rigcheck_OBJS := host/vt_rigcheck host/rigwatch
vt_stars_OBJS := host/vt_stars host/stars
vt_mapstat_OBJS := host/vt_mapstat
vt_mapcheck_OBJS := host/vt_mapcheck host/calib
vt_attitude_OBJS := host/vt_attitude host/calib host/stars kern/attitude kern/depth kern/img_ref kern/img_vx
vt_agree_OBJS := host/vt_agree host/calib host/stars kern/depth kern/img_ref kern/img_vx

all: host card
host: $(HOST_PROGS:%=out/host/%)
card: $(CARD_PROGS:%=out/card/%)

out/card/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(KNC_CLANG) $(CARD_CFLAGS) -c $< -o $@

out/card/%.o: src/%.S
	@mkdir -p $(dir $@)
	$(KNC_CLANG) $(CARD_CFLAGS) -c $< -o $@

out/host/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(HOST_CC) $(HOST_CFLAGS) -c $< -o $@

define card_prog
out/card/$(1): $$($(1)_OBJS:%=out/card/%.o)
	$$(KNC_LINK) -static -o $$@ $$^ -lpthread -lm
	@scripts/audit.sh $$@
endef
$(foreach p,$(CARD_PROGS),$(eval $(call card_prog,$(p))))

define host_prog
out/host/$(1): $$($(1)_OBJS:%=out/host/%.o)
	$$(HOST_CC) -o $$@ $$^ -lm -lpthread $$($(1)_LIBS)
endef
$(foreach p,$(HOST_PROGS),$(eval $(call host_prog,$(p))))

check: all
	@scripts/check.sh

compdb:
	@scripts/compdb.sh

clean:
	rm -rf out/card out/host

.PHONY: all host card check compdb clean
-include $(shell find out/card out/host -name '*.d' 2>/dev/null)
