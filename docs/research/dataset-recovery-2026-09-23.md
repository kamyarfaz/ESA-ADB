# Test CSV integrity investigation — 23 September 2026

**Current status: RESTORED AND VERIFIED.** The canonical reconstruction matched
the exact previously verified whole-file SHA256. Its independent disk read and
the installed-file read both matched too. The damaged original is preserved at
`data/repairs/2026-09-23/84_months.test.damaged.csv`. The data gate is cleared.

## Earlier failed candidates (historical)

At this earlier stage no replacement had been installed. The seven-correction candidate
still failed the whole-file known-good checksum. Preserve original and candidate.
A subsequent full byte comparison showed an additional difference at offset
2373559484 (original 119, candidate 55), outside the seven intentional edits.
The earlier invalid-byte scan did not report this offset. This inconsistency
requires further source/copy/read-integrity investigation; cause is not established.
The experiment remained blocked at that stage; subsequent verified recovery is
recorded below.

The longer-run preflight found that the train CSV still matched its verified hash,
but the test CSV hash was `1f783660d322fbafef0054315d62bcb3a21befa1284b8a1bcd0f3fdc2e75f42f`.
The known-good test SHA256 is
`9990c46dba0101074a19b7e58382bee787ea9a2051230a5b1716d207d8315f6c`.

A full timestamp scan found one malformed timestamp, 2009-04-09 19:15:00.
A byte scan found six invalid bytes in one small region. A timestamp-only candidate
and a six-byte candidate both failed the golden checksum and were not installed.
A raw-to-CSV comparison around this region found one additional valid-looking
numeric change on channel 59 at 19:17:30: 0.91505065 instead of 0.91505075.

Candidate corrections (byte offsets, decimal byte values):

| Offset | Damaged | Correct |
| --- | ---: | ---: |
| 2373546556 | 172 | 44 |
| 2373546684 | 14 | 46 |
| 2373553340 | 16 | 48 |
| 2373554748 | 37 | 53 |
| 2373556284 | 144 | 48 |
| 2373558972 | 114 | 50 |
| 2373558844 | 54 | 55 |

Evidence includes timestamp-grid constraints, comma/decimal syntax and independent
raw zero-order-held values for channels 47, 71 and 59. The final authority is an
exact match of the entire candidate file to its previously verified SHA256.
No candidate may replace the active input without that whole-file match.

Local recovery scripts, scans and the preserved damaged original are under
`data/repairs/2026-09-23/`. Do not blindly rerun these one-time scripts. The compact
verification reports are copied to `tables/dataset-recovery-2026-09-23/` after
successful installation. This record does not diagnose the cause of corruption;
repeated unexplained changes warrant server memory/storage/background-writer
investigation. A successful hash check verifies the current snapshot only.

A later independent checksum of the active original was
`4e6f4c43a3e97cdfa22cda05393e32b6c1bd80b631d40c7ae6c84335662cef21`,
different from the initial read despite no active-file write by this task.
Read stability/background modifications/storage or memory must be investigated.
This is evidence of inconsistent input reads, not a diagnosed hardware cause.

## Follow-up investigation

Two further full reads of both original and candidate produced stable hashes and
unchanged size/mtime/ctime during those reads. This does not explain the earlier
change. The candidate passed all 7,364,161 rows: 175 columns, exact 30-second
timestamps, finite numeric values and label codes. No invalid characters remain.

A full independent reconstruction of all 87 feature streams and paired labels
from the raw sources, following the existing preparation transformations, found
zero numeric mismatches against the candidate's float32 features and uint8 labels.
This includes differencing, forward hold, annotation restoration and telecommand
encoding. Float32 equivalence does not prove byte-identical numeric text. A separate
reserialization attempt is therefore being checked against the golden file hash;
no replacement is authorized by numerical equality alone.

## Successful restoration

The final reconstruction serialized all independently checked float32 feature
values and uint8 label values, with the exact timestamp grid, to a separate CSV.
Its complete SHA256 was
`9990c46dba0101074a19b7e58382bee787ea9a2051230a5b1716d207d8315f6c`,
matching the historical golden dataset. The written file was independently hashed
before installation. Only then was the original renamed to a preserved backup
and the verified reconstruction installed. A further installed-file read matched
the same hash. No new dataset version or changed experiment values were accepted.

The cause of the earlier corruptions/read discrepancy remains undiagnosed. This
restoration verifies the current snapshot; the long-baseline command independently
checks both dataset hashes each time it starts or resumes. Raw annotations were
used solely to reconstruct existing labels for integrity verification, not to
select models or inspect performance.

The final installed/candidate comparison isolated the last remaining difference:
at byte offset 2373546172, the candidate had space (32) where the golden file has
zero (48). Numeric parsing accepted this without a float32 mismatch. This explains
why numeric agreement alone was insufficient. The actual `long-baseline
--verify-data` preflight subsequently passed for both installed CSVs.
