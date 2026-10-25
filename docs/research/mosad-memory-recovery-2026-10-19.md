# MOSAD operational memory recovery

**Historical operational record. MOSAD was subsequently stopped at the user’s
request on21 October2026. Recovery instructions below are superseded and must
not trigger a restart. Read AI_HANDOFF.md for the current state.**

**19 October memory-bounded recovery active (supersedes waiting note below).**
User explicitly requested resuming with less RAM and safeguards. Same service
restarted as PID3497287 with unchanged Python source, data, scientific settings,
and original --resume validation. Operational drop-in:
`~/.config/systemd/user/esa-similarity-mosad.service.d/memory.conf` sets
MALLOC_ARENA_MAX=2, MALLOC_TRIM_THRESHOLD_=0, MemoryHigh=4G,
MemoryMax=4608M, MemorySwapMax=256M; Restart=no remains in force.
Frozen checks and full training-data loading passed. GPU activity confirmed for
this PID (2070MiB VRAM); service memory current3.15GiB, peak3.58GiB, swap0
at the initial check. Resume uses saved epoch3 batch1 state; next batch log at100
is still pending. Do not mistake old batch1 log for new progress. These limits
reduce shared-server pressure but cannot guarantee survival under host OOM or
later calibration peaks. On failure inspect cause; do not blindly restart or
raise the cap. Monitor this bounded recovery instead of waiting for8GiB free.
A separate preserved checkpoint copy is in
`/home/k_faz/project-backups/run-logs/mosad-before-memory-recovery-20261019.pt`,
SHA256 e61952b0e728ac601bcfed61764643d6a42f21bc7ca465a175e2e6e64eb26192
(equal to the original before recovery). No unrelated processes were stopped.


Allocator and cgroup settings affect resource management; model, batch size, sequence length, sample ordering, optimizer, RNG recovery, splits and threshold protocol remain unchanged. Original protocol files were preserved. Subsequent stage completion and final independent verification remain required.

**20 October00:00 CEST check: bounded recovery interrupted again.**
At19October22:52:45 CEST PID3497287 exited by SIGKILL, Result=signal,
not the earlier explicit oom-kill result. Cause is unconfirmed: this account
cannot read system/kernel/other-user journals. Service peak3.58GiB and swap8MiB
were below configured caps; no Python error appears. No runner or CUDA compute
process remained. Latest checkpoint saved19October22:37:16 at epoch3 batch1100.
At inspection only3.8GiB RAM was available (swap2.6GiB free), leaving almost no
startup headroom above the measured peak. Restart deferred; next heartbeat
reassess headroom (prefer at least5GiB available for the4.5GiB-capped service),
check no worker, and resume the same bounded service if safe. Do not assume the
SIGKILL was caused by the cap; do not raise limits or change frozen source.
If it recurs despite sufficient memory, administrator logs are needed to identify
the killer. Existing checkpoints and pre-recovery backup remain preserved.


**20 October03:14 CEST recovery restarted.** Available RAM5.3GiB,
swap4.7GiB; no comparison worker remained. Same bounded service restarted as
PID3880307 with original --resume and unchanged source/data/settings. Frozen
checks and full training input loading passed; GPU activity confirmed for this
PID (2062MiB VRAM). Initial memory3.57GiB, swap0, peak4.00GiB during
startup. Latest saved state remains epoch3 batch1100; await new batch1200
checkpoint to confirm further persisted progress. Original unexplained SIGKILL
cause remains unresolved. Keep existing4.5GiB cap and no blind restart policy.

