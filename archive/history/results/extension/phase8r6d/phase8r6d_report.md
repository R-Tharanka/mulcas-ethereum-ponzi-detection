# Phase 8R6D Final Report

## Scope And Integrity

Phase 8R6D resumed the frozen provider ledger, completed external static-data auditing where provider health allowed, and froze the prospective Phase 9B protocol. No detector was trained or scored, P5 was not evaluated, Track B labels were not used for development, and Track A was not modified.

## Enrichment

- Candidates: 1490
- Attempted: 1490
- Successful: 1453
- Invalid contracts: 34
- Identity mismatches: 2
- Unsupported schemas: 0
- Creation metadata not found: 1
- Provider failures: 0
- Provider requests in Phase 8R6D / cumulative: 4090 / 8930
- Provider retries in Phase 8R6D: 0
- Additional runtime: 26.80 minutes
- Cumulative runtime: 74.52 minutes

Both identity mismatches were isolated receipt-contract-address mismatches. The
creation-provider identity matched the requested address and transaction/block
resolution succeeded, but strict receipt identity validation failed. Both rows
remain excluded from Track B.

## Feature Alignment

MULCAS_READY: 1453. STATIC_RUNTIME_READY: 1453. CREATION_CODE_READY: 1453.

## Track B

Later positives/benigns: 112 / 180. Strict Track B positives/benigns: 112 / 180. Provisional positives/benigns: 0 / 0. Rating: **STRONG**. Sealed: **True**. Critical manual reviews: 0.

## Decision

**A. TRACK B READY + PHASE 9B FROZEN**

Next recommended phase/action: `phase9c`. Phase 9C was not started.
