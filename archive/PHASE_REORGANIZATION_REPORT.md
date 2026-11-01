# Historical Archive Phase Reorganization

**Date:** 2026-11-01  
**Scope:** `archive/history/` only

## Purpose

The previous archive mirrored the original repository tree. That preserved
provenance but mixed phase-owned files under shared directories such as
`configs/`, `src/`, and `results/`. The archive is now physically ordered by
research chronology.

## New Layout

```text
archive/history/phases/<chronology group>/<original repository path>
```

Seven chronological packages are present:

| Package | Files relocated |
| --- | ---: |
| `00_05_foundation_and_baseline` | 66 |
| `05_5_paper_aligned` | 15 |
| `06_phase6_improvements` | 67 |
| `07_phase7_analysis` | 55 |
| `08_phase8_extension` | 293 |
| `09_phase9_prospective` | 50 |
| `10_cross_phase_synthesis_and_support` | 21 |
| **Total** | **567** |

Each package keeps original relative paths inside it. This means a file can be
restored to its original location without guessing its former directory.

## Phase Ownership Rule

Files belong to the phase that created or consumed them, not necessarily the
phase number appearing in the filename. For example:

- `configs/phase7/temporal_robustness.yaml` belongs to Phase 7.
- `configs/extension/frozen_phase7_manifest.yaml` belongs to Phase 8 because it
  freezes Phase 7 as Phase 8's predecessor input.

This rule removes the apparent Phase 7 duplication while preserving the actual
research dependency chain.

## Integrity Verification

- Manifest entries checked: **567**
- Missing archived files: **0**
- Duplicate archived destinations: **0**
- Non-LFS Git blob mismatches: **0**
- Materialized LFS exceptions: **1**, expected and verified
- LFS object size: **107,441,279 bytes**
- LFS object SHA-256:
  `c58dede57cdc88b852f862f01769eb5393b26da8e03ba16623c77b0cf4f1a269`

No experiment result, configuration content, source implementation, test, or
historical conclusion was changed. Only archive locations, navigation files,
and archive-path references were updated.

## Navigation

- `archive/CHRONOLOGY.md`: chronological overview
- `archive/history/README.md`: package usage and restoration rules
- `archive/history/TRACKED_FILE_MANIFEST.csv`: authoritative file lookup
- each phase package's `README.md` or `PHASE_GUIDE.md`: phase-specific guide
