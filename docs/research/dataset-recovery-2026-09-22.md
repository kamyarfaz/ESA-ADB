# Dataset recovery — 22 September 2026

The specialist run failed with a UTF-8 decoding error before its first training
epoch. Its 2003 output directory was empty. No model checkpoint was produced.
The failed input SHA256 was
`2b49aeba797d9a08f540c617668b3874a9249d44f1da1c626432605810847879`,
which differs from the previously verified dataset hash.

A whole-file comparison found six differences versus the preserved source:
749580860: 184→56; 749585468: 119→55; 749592764: 40→44;
749594684: 142→46; 749606076: 113→49; 749606972: 119→55.
Values are decimal byte values. They include two non-ASCII bytes and a changed
comma. An initial two-byte candidate failed the full hash check and was not used.

The preserved source at `data/repairs/2026-09-16/84_months.train.damaged.csv`
currently had the exact known-good hash, despite its historical filename. Both
historically repaired offsets already contained the corrected values; the recovery
copy therefore made no additional changes to that source. Its current contents,
not the old filename/report, determined its suitability for recovery.

The active CSV was replaced only after the replacement matched the full known-good
SHA256 `fe6812dd5d324fe2eb347a703894551f0a03f40253742a3cd35f42d02fdb80ad`.
The failed active file is preserved in `data/repairs/2026-09-22/84_months.train.damaged.csv`.
Other local reports and scripts in that directory preserve the recovery evidence.
No encoding fallback, ignored errors or skipped malformed rows were introduced.

The cause is unknown. Repeated unexplained content differences warrant checking
server memory/storage and background writers with the server administrator; this
report does not diagnose a particular hardware fault. The earlier successful
checksum was a snapshot, not a guarantee that files could not change later.
Do not claim all unrelated artifacts were revalidated by this dataset recovery.

Use a fresh specialist output directory `ae_specialist_seed42_verified` because
input metadata changed; preserve the original failed-run protocol. The combination
command must use the same new specialist root. Run under tmux to survive logout.
No GPU training was launched by the assistant during recovery.

A fresh independent read confirmed the full checksum after replacement. The exact
specialist first-fold loader then read 3,156,480 rows successfully. No training
was performed. [Verification records](tables/dataset-recovery-2026-09-22) are versioned.
