# Phase 8R6B: external enrichment and Track B construction

## 1. Purpose

Phase 8R6B constructs and audits external evaluation data; it does not measure detector performance.

## 2. Scope

Deployment/static enrichment only. No transaction behavior, model training, detector scoring, P5 evaluation, or Phase 9B execution occurred.

## 3. Frozen artifact verification

Verified 28 Phase 0-8R6A anchors and 5218 downloaded raw files by SHA-256 before construction.

## 4. Phase 8R6A starting candidates

Primary ponzi_ml list: 537 positive and 927 benign new addresses. Adding 65 new Blockchain@Unica positives, with 39 already in ponzi_ml, yields 1490 unique addresses and 563 positive labels. Source records remain separate.

## 5. Enrichment provider/method

Blockscout's credential-free contract-creation API supplied creation transaction, creator, block, timestamp and creation bytecode. PublicNode Ethereum JSON-RPC supplied runtime code, transaction, receipt and block; address/block/timestamp consistency was required.

## 6. Performance preflight

The required 10-positive/10-benign sample took 13.341 seconds and initially projected 16.57 minutes, with 9/20 full-field successes. The subsequent run received HTTP 429 for all 8 attempted rows; retry backoff alone projected 149.0 minutes, so execution stopped at the 30-minute gate.

## 7. Enrichment completion rate

Successfully enriched 0 candidates. Provider-rate-limit failures remain in the resumable ledger with attempts/provider/error; successful rows would not be repeated on restart.

## 8. Creation block/time findings

Verified later counts are 0 positive and 0 benign. Block 7,136,486 is primary; timestamp is a consistency check.

## 9. Runtime/opcode/creation overlap

Runtime was fetched from chain, opcode was locally derived with the frozen repository disassembler, and creation input was taken from the verified creation transaction. Missing creation-code data are treated as uncertainty, not evidence of non-overlap.

## 10. Cross-source provenance/conflicts

There are 30 unresolved conflicts across all external sources and 0 among construction candidates. Class-D disagreements are resolved only in favor of an existing A/B record and never upgrade predictions. Agreement does not erase source-specific provenance.

## 11. ponzi_ml label audit

ponzi_ml describes manual source-code criteria and incorporates earlier corpora, but benign sampling/matching and per-row verification detail remain under-specified. Labels remain class B and are not overstated as class A.

## 12. Blockchain@Unica reference subset

Blockchain@Unica remains a separate class-A positive reference. Shared addresses retain both provenance records and are not blindly duplicated.

## 13. Benign quality audit

ponzi_ml benign rows are BENIGN_CURATED, not BENIGN_STRONG. Random/matched sampling is under-specified and unlabeled-scam risk cannot be excluded; the row-level audit is retained.

## 14. Later-contract counts

Later positives: 0; later benigns: 0. Publication date was never used as deployment time.

## 15. Strict-independence counts

Strict code-new counts: 0 positive and 0 benign. Provisional counts: 0 and 0. New address, new code, and later deployment are separate properties. A contract is not treated as temporally independent merely because its address was absent from the original corpus.

## 16. Manual-review triage

Generated 31 M1-M5 triage rows. M5 model predictions are never labels; critical groups above 30 receive a plan rather than automatic review.

## 17. Track B construction

Primary membership applies all ten frozen rules and contains 0 unique addresses. Model-generated Ponzi detections remain excluded from the primary Track B ground truth unless independently verified.

## 18. Track B composition

Positive=0, benign=0; no random rebalancing was performed.

## 19. Track B duplicate audit

The primary view retains one row per address. Exact code duplicate counts and the deterministic STRICT_UNIQUE_CODE_SENSITIVITY policy are recorded without inspecting performance.

## 20. Sealed-set readiness

Rating=NONE; SEALED_READY=no. No sealed benchmark was created.

## 21. Portability maintenance fix

Resolved and hash-verified 27 historical checkpoint/embedding records using only approved repository-relative roots. Frozen manifests were not edited and hashes were not weakened.

## 22. Limitations

The credential-free creation provider imposed HTTP 429 rate limits during the full run. Failed rows are not inferred pre-existing, later, or independent. Exact hashes do not measure fuzzy similarity. Completion needs slower scheduled batches or a reputable credentialed provider.

## 23. Reproducibility/tests

Configuration freezes providers, retry policy, boundaries, normalization, provenance, conflicts, membership, rating, sealing, and no-model rules. Environment and provider/API classes are recorded in the preflight artifact.

## 24. Recommended next phase

B. PHASE 8R6C MANUAL VERIFICATION / BENIGN REFINEMENT FIRST. Do not begin Phase 9B automatically.
