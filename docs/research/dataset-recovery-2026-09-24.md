# Diagnostic CSV recovery — 24 September 2026

The user completed 12,000 training-window inferences, then calibration parsing
failed with invalid UTF-8 byte 0xb2. The initial diagnostic SHA check had passed.
A subsequent active CSV read hashed to 741aee07c1db98c36ef822af3c888d6f4c5c99fa27447546142ff82e98e4f526,
not the expected golden hash. Five byte differences were found in offsets
4320116924–4320130236, including both invalid encoding and valid ASCII changes.
Do not switch encodings or skip bad lines.

A candidate reconstructed from the preserved September 16 backup matched the
original golden SHA. The two historical replacement positions already held their
expected bytes on this read (no change was needed); do not assume the previously
named damaged backup is consistently damaged or consistently healthy. The cause
of these differing observations is unknown. Repeated incidents merit checking
server RAM, storage, filesystem and kernel logs with the administrator.

An independent complete candidate read matched the golden SHA before installation.
The active file was archived to data/repairs/2026-09-24/84_months.train.before_restore.csv.
The verified candidate replaced it. The exact calibration loader then successfully
read 1,051,200 rows by eight channels for 2006. A subsequent complete installed-file
hash again matched fe6812dd5d324fe2eb347a703894551f0a03f40253742a3cd35f42d02fdb80ad.

Original diagnostic v1 and its training cache remain preserved, but the retry must
use ch40_47_epoch10_v2. A finite cache does not prove its source was intact during
loading. No training or model checkpoint was changed. Diagnostic code now performs
an end-of-run full CSV hash check as well as initial verification. All 47 tests pass.
No claim is made that the underlying server integrity problem has been resolved.

Compact evidence: tables/dataset-recovery-2026-09-24/.
