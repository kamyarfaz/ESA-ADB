# Historical calibration coverage

Read-only annotation and frozen-rule review, Jan–Mar of each year. No assessment
outputs accessed. Tables: `tables/calibration-coverage-2026-09-26/`.

| Year | System events | Events annotated on inputs40–47 | System TP | FP | FN |
|---|---:|---:|---:|---:|---:|
|2003|3|1|1|0|2|
|2004|1|1|1|2|0|
|2005|4|3|2|0|2|

Aggregate TP counts do not establish which event IDs were detected. In2005 at
least one of3 locally annotated events must have been missed because only2
system events were detected. Fresh frozen-checkpoint calibration diagnosis is
required to identify it. Event id_10 extends past the calibration endpoint;
id_12 in2004 begins before calibration. Counts are sparse and durations differ.

The existing2004 diagnostics show the real alarm peak3.55125 onchannel47, while
false-alarm peaks3.64783 and3.62879 exceed it. Raising a scalar threshold cannot
retain this real alarm while eliminating both false alarms. Choosingchannel47
alone from this single positive would risk fitting the calibration example.
