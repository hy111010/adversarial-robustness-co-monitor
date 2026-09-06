# Supplemental execution audit

- 2026-09-04: The first fixed-epoch-21 controls were excluded before paper
  analysis because the detector callback was still permitted to intervene.
  This was visible in their recorded trigger epochs (17, 21, and 13 rather
  than 21 for every seed).
- The implementation was corrected so fixed-schedule controls retain the same
  trace instrumentation but ignore adaptive alarms. All three fixed-epoch-21
  runs are being repeated under the preregistered schedule.
- Excluded folders are preserved with the suffix
  `_invalid_detector_active`; they are not read by the summary script.
- The fixed-epoch-13 runs all switched at epoch 13 before any adaptive alarm
  and therefore already implement the intended trajectory. They are retained.

