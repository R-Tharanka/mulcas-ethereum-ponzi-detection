# Paper Reproduction Notes

## Phase 5.5 Protocol Freeze

Phase 5.5 is a correction of the independent baseline protocol, not a model
improvement experiment. All Phase 0-5 outputs under `results/baseline/` remain
historical records. Corrected outputs are isolated under
`results/baseline/paper_aligned/`.

The corrected chronological development protocol is:

- training: contracts through the 200th chronological Ponzi contract;
- validation: later contracts through the 250th chronological Ponzi contract;
- final test: later contracts through the 314th Ponzi contract;
- benign contracts follow the same creation-block boundaries; and
- contracts are never shuffled.

Whole-block assignment is used at boundaries so contracts created in the same
block cannot appear in different partitions.

For the released CSV, the measured boundaries are block `5,021,809` and block
`5,038,334`. This produces 5,839 training rows, 150 validation rows, and 509
test rows. Combining training and validation gives 5,989 rows through the 250th
Ponzi contract, while the publication reports 5,990 training rows and 508 test
rows for its two-way split. The publication does not explain this one-row
difference, so it remains documented rather than forced to match.

## Feature Corrections

The prior 70/15/15 baseline's unfiltered unigram/bigram/trigram implementation
has 70,867 dimensions and is retained unchanged. The corrected representation
uses unigrams, bigrams, and trigrams with training-only document-frequency
filtering. The publication describes filtering low-document-frequency N-grams
but does not report the exact cutoff. Phase 5.5 therefore freezes `min_df = 2`
as an explicit reconstruction assumption, not as a tuned value.

The publication describes each contract's Word2Vec representation as an
IDF-weighted sum of 100-dimensional opcode embeddings. Phase 5.5 implements
that representation while preserving the existing unweighted mean as a named
alternative feature variant.

The chronological creator indicator is appended to every static opcode feature
representation. Training rows use strictly earlier blocks, while validation and
test rows use creator state frozen after training without reading later labels.
It is not treated as an unrelated sixth paper viewer in the corrected runs.

## Published Metric Inconsistency

Table 5 reports MulCas recall `0.674`, precision `0.951`, and F1 `0.789`.
Nearby prose states that MulCas achieved F1 `0.743`. These values conflict.
This project uses the Table 5 value for tabular comparison because it appears
with the corresponding precision and recall, but preserves the prose value as
an unresolved publication inconsistency. It is not silently corrected.

## Remaining Publication Gaps

The publication does not fully specify the N-gram document-frequency cutoff,
all classifier hyperparameters and software versions, Word2Vec training details
beyond the 100-dimensional representation, tie handling at creation-block
boundaries, or a separate validation-set protocol. Phase 5.5 records every
chosen value in YAML and does not tune any choice against final-test labels.
