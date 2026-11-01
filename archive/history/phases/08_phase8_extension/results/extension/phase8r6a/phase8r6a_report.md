# Phase 8R6A: external Track B qualification

## 1. Purpose

Phase 8R6A evaluates dataset independence and suitability; it does not evaluate detector performance.

## 2. Scope

The work is limited to acquisition, normalization, exact hashing, descriptive overlap, provenance, and feasibility. No model was trained, tuned, scored, or used to evaluate P5.

## 3. Frozen prior-artifact verification

Verified the 9 Phase 9A freeze anchors plus 210 recursively checked Phase 0-8 manifest events, 9 checkpoints, and 18 embedding artifacts before analysis. No prior result was modified.

## 4. Candidate-source registry

Eight separately named candidates/views were retained: the XBlock/current reference, Blockchain@Unica, CTRF, ponzi_ml, PonziGuard's manually verified 50, its 805 detector predictions, its 31 later predictions, and its address-unresolved ground-truth archives. Sources were not silently merged.

## 5. Data acquisition status

Official repositories were acquired below the 5 GB/30-minute gates. The manifest records 5218 non-git files with canonical URL, deterministic retrieval timestamp, byte size, and SHA-256. No blockchain-wide lookup was launched.

## 6. Label provenance

Blockchain@Unica and PonziGuard's verified 50 are class A; CTRF and ponzi_ml are class B; both PonziGuard prediction lists are class D; the current XBlock corpus is reference-only. Model-generated suspected Ponzi contracts are not treated as independent ground truth unless separately validated.

## 7. Address overlap

Unique-address overlap/new counts are: Unica 117/65; CTRF 3,721/163; ponzi_ml 2,909/1,464; PonziGuard verified 41/9; PonziGuard predictions 165/640; later predictions 0/31. The XBlock candidate is the current corpus itself (6,483/0). All 131 external/internal label-conflict rows are preserved explicitly.

## 8. Runtime/opcode/creation-code overlap

CTRF supplied 3,751 usable runtime hashes (7 overlap) and 3,291 opcode hashes (0 overlap). ponzi_ml supplied 1,038 usable runtime hashes (203 overlap); 14 new addresses duplicate internal runtime code. Creation-code overlap is unavailable externally. Missing fields are not interpreted as non-overlap.

## 9. Creator overlap

No audited external release supplied creator addresses in its downloadable label table. Creator statistics are therefore explicitly not computable and require chain enrichment.

## 10. Cross-source overlap

A newer publication does not imply an independent dataset. Independence is established only after address/code overlap and label provenance are audited. CTRF and ponzi_ml overlap by 2,921 addresses; Unica is wholly represented in CTRF at 182 unique addresses; all 50 verified PonziGuard addresses occur in its prediction list.

## 11. Temporal novelty

No address-resolved source supplied complete creation blocks or timestamps. Therefore genuinely later positive count is unknown, not zero. Enrichment would require deterministic creation-transaction/block lookup for candidate addresses; public bulk RPC was not assumed free or reliable within the gate.

## 12. Positive/benign availability

After address-overlap removal: Unica has 65 positives/0 benign; CTRF 162/1; ponzi_ml 537/927; verified PonziGuard 9/0. Detector-output counts (640 and 31 new suspected positives) remain class D and do not contribute ground truth.

## 13. Candidate quality subsets

S0 and S1 are computed for every address-resolved source. S2/S3 are reported only when required code fields cover all rows; S4 cannot be fully verified because external creation code is absent; S5 cannot be computed without deployment time. Missing-code rows are retained.

## 14. Sampling/bias considerations

Principal biases include hand-curated positive lists, verified-source availability, transaction-history-dependent discovery, detector preselection, random/curated benign sampling, and period mismatch. These limitations are documented rather than used as automatic rejection rules.

## 15. Manual verification feasibility

The 31 later PonziGuard predictions form a bounded 31-100 review tier, while the 640 unverified new detector predictions exceed 100 and call for staged review. The nine new manually checked hits are already class A but remain positive-only. ponzi_ml requires conflict resolution plus time/code enrichment, not wholesale relabeling by this phase.

## 16. Track B feasibility ratings

Best threshold-qualified candidate: blockchain_unica, MODERATE feasibility, with 65 new positive addresses. It is positive-only and not a complete benchmark. ponzi_ml has 537 new positive and 927 new benign addresses but cannot receive STRONG under the frozen rules until temporal metadata and strict overlap are resolved.

## 17. Sealed-set readiness

No candidate is sealed. Membership is not frozen because temporal/strict-overlap enrichment remains outstanding. Phase 9B-9F must not use external labels before final membership and hashes are frozen; no exploratory model scoring was performed.

## 18. Limitations

No source supplies complete creation block/timestamp metadata. Strict S4/S5 identities cannot be claimed where code fields are absent. Archive contract names could not be fabricated into Ethereum addresses. Exact hashing detects clones only under the declared normalization and performs no fuzzy similarity.

## 19. Reproducibility/tests

Configuration freezes normalization, source registry, cutoffs, taxonomy, thresholds, directories, processing gates, and the no-model constraint. Dedicated tests cover hashes, determinism, duplicate/conflict preservation, label eligibility, positive-only governance, outputs, and absence of model/P5 evaluation.

## 20. Recommended next phase

B. EXTERNAL DATA ENRICHMENT / MANUAL VERIFICATION FIRST. Enrich creation block/timestamp and missing exact-code identities, resolve listed conflicts without silent relabeling, construct or validate benign controls, then freeze membership and hashes. Do not begin Phase 9B automatically.

No suitable sealed Track B benchmark was established from the audited public sources. This does not mean such a benchmark cannot be constructed; it means the currently audited public data are insufficient under the predefined criteria.
