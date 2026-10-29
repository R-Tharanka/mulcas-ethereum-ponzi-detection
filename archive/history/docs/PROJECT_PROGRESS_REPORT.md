# MulCas Ethereum Ponzi Detection: Project Progress Report

## 1. Purpose and Current Status

This project rebuilds and tests MulCas, a system for finding Ethereum smart
contracts that are labelled as Ponzi schemes. It uses only information that
would have been available when each contract was created. The work is based on
the released dataset and the paper *Securing the Ethereum from Smart Ponzi
Schemes*.

In simple terms, the project does four things: it checks the source data, turns
contract bytecode into numbers, trains several machine-learning models, and
tests whether the models still work on contracts created later in time.

Work is complete through **Phase 7C**:

- Phases 0-5 built the data preparation, feature extraction, model training,
  and MulCas testing pipeline.
- Phase 5.5 changed the data split to match the paper more closely. Its result
  became the fixed reference result for later comparisons.
- Phases 6A-6E tested specific possible improvements. None was reliable enough
  under the time-based testing rule, so Phase 6F was not run.
- Phase 7A tested performance on different time periods. Phase 7B removed
  MulCas parts one at a time. Phase 7C removed the creator-history feature.
- Phase 7D, Phase 7E, and final evaluation notebook 08 have not been performed.

The main conclusion is that none of the tested changes reliably improved the
reference model. The model works very differently in different time periods:
it performs well in some periods and poorly in others. This does not prove that
improvement is impossible; it shows that future improvements must be tested
carefully on later data.

## 2. How to Read the Metrics

- **Precision:** when the model says “Ponzi”, how often it is correct. Low
  precision means many false alarms.
- **Recall:** of all real Ponzi contracts, how many the model finds. Low recall
  means many Ponzi contracts are missed.
- **F1:** one number that combines precision and recall. A high F1 requires a
  reasonable balance between both.
- **PR-AUC:** a ranking score. It checks whether real Ponzi contracts are
  generally ranked above benign contracts over many possible cut-off points.
- **Balanced accuracy:** the average of Ponzi recall and benign recall. It is
  useful here because benign contracts greatly outnumber Ponzi contracts.
- **TN / FP / FN / TP:** four counts used to describe predictions: correctly
  rejected benign contracts, wrongly flagged benign contracts, missed Ponzi
  contracts, and correctly found Ponzi contracts.

Other words used often in this report:

- **Feature:** a numeric description of a contract used by a model.
- **View:** one type of contract description, such as opcode counts or TF-IDF.
- **Opcode:** a low-level instruction in Ethereum bytecode.
- **Temporal / chronological:** ordered by creation time.
- **Data leakage:** using information that would not have been available when
  the prediction was made.
- **Frozen baseline:** the fixed reference model used for fair comparisons.
- **Ablation:** removing one part of a system to measure its effect.
- **CV:** cross-validation, meaning repeated train-and-check tests inside the
  historical training data.
- **TF-IDF:** a way to give more weight to useful terms that are not common in
  every contract.
- **N-gram:** a sequence of neighbouring opcodes. A 1-3-gram contains sequences
  of one, two, or three opcodes.
- **Word2Vec:** a method that converts opcode words into numeric vectors.

The dataset is highly imbalanced: there are many more benign contracts than
Ponzi contracts. For example, a model could achieve high ordinary accuracy by
calling almost everything benign while missing most Ponzi contracts. For this
reason, the report focuses on precision, recall, F1, PR-AUC, balanced accuracy,
and the four confusion-matrix counts together.

## 3. Data and Reproducibility Foundation

The immutable input is `data/raw/PonziDataset_20221114.csv`.

| Property | Measured result |
| --- | ---: |
| File size | 307,176,676 bytes |
| SHA-256 | `48b13eab1f87c68b3e486f5932df22c7907e76bd23b4042de8d0b2a83ce846cf` |
| Contracts | 6,498 |
| Ponzi contracts | 314 |
| Benign contracts | 6,184 |
| Earliest block | 49,880 |
| Latest block | 7,136,486 |
| Earliest timestamp | 2015-08-07 20:48:27 |
| Latest timestamp | 2019-01-27 23:52:51 |
| Unique creators | 3,360 |

The released file contains 6,498 contracts, including 314 Ponzi contracts, so
it matches the paper's headline totals. The audit found no empty required
values and no empty opcode strings. It did find repeated addresses,
transaction hashes, block/address pairs, and identical opcode strings. Those
rows were reported and kept. They were not silently deleted because changing
the source data would make the reproduction less faithful.

The following rules make it possible to repeat the work and check that future
results were produced in the same way:

- repository-relative paths and deterministic random seeding;
- an isolated `.venv` and explicit dependencies;
- chronological ordering by `createdBlockNumber`;
- train-only fitting of vocabularies, IDF statistics, Word2Vec, scalers, and
  estimators;
- creator features based only on labels of strictly earlier contracts;
- compact split-membership files instead of copied 307 MB datasets;
- machine-readable configurations, metrics, traces, hashes, and frozen-phase
  manifests;
- large generated matrices and fitted models kept under ignored artifact paths.

Detailed audit: `results/baseline/dataset_audit.json` and
`docs/dataset_description.md`.

## 4. Work Completed Through Phase 5.5

### Phase 0 - Local Research Environment

The project was made easy to run from Python. Common path and random-seed
helpers were added, basic tests were added, generated files were excluded from
Git, and Windows setup instructions were documented. This phase created the
foundation for the later experiments; it did not produce a model score.

### Phase 1 - Dataset Integrity Audit

The raw CSV was read without changing it. The audit checks what columns exist,
how many rows and labels there are, whether values are missing, whether rows
are repeated, whether dates and block numbers are valid, whether creators are
reused, whether opcode text is usable, how much memory is needed, and whether
the file fingerprint is unchanged. It avoids loading large text columns when
they are not needed.

### Phase 2 - Initial 70/15/15 Chronological Split

The first reconstruction divided contracts into training, validation, and test
groups by creation order. Older contracts were used first and rows were never
shuffled. This imitates the real use case: train on known historical contracts
and predict newer contracts.

| Split | Rows | Ponzi | Benign | Benign:Ponzi | Block range |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 4,548 | 140 | 4,408 | 31.49:1 | 49,880-4,572,153 |
| Validation | 974 | 8 | 966 | 120.75:1 | 4,572,291-4,955,839 |
| Test | 976 | 166 | 810 | 4.88:1 | 4,957,244-7,136,486 |

The groups contain different rows and addresses, but some identical code text
appears in more than one group. The project measured and reported this instead
of hiding it. Therefore, the split is time-ordered, but it is not a guarantee
that every repeated piece of code appears in only one group.

### Phase 3 - Independent Feature Reproduction

The bytecode cannot be given directly to the models, so it was converted into
the following numeric views. The tools that learn these views used training
contracts only:

| Feature view | Dimension | Meaning |
| --- | ---: | --- |
| Opcode term count | 142 | Counts of individual opcodes |
| Opcode TF-IDF | 142 | Term counts reweighted by training IDF |
| Opcode 1-3-gram count | 70,867 | Counts of opcode sequences of length 1-3 |
| Opcode 1-3-gram TF-IDF | 70,867 | N-gram counts reweighted by training IDF |
| Mean Word2Vec | 100 | Mean learned opcode embedding per contract |
| Creator feature | 1 | Leakage-safe prior-Ponzi-creator indicator |

This is important: validation and test contracts were only converted using
rules learned from training contracts. They were not allowed to change the
vocabulary, weights, or embeddings.

### Phase 4 - Individual Baseline Models

Three model types were tested: Ridge, linear SVM, and XGBoost. Each was tested
with the different feature views. The first validation group contained only
eight Ponzi contracts, so choosing a model from that group was unstable. The
table shows the feature view selected from validation and the resulting test
score.

| Model | Selected original view | Val P | Val R | Val F1 | Test P | Test R | Test F1 | Test PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | N-gram TF-IDF | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6160 |
| SVM | N-gram count | 0.2500 | 0.5000 | 0.3333 | 0.5897 | 0.1386 | 0.2244 | 0.3509 |
| XGBoost | Term count | 1.0000 | 0.3750 | 0.5455 | 0.0000 | 0.0000 | 0.0000 | 0.4540 |

The first results were weak and sensitive to the small validation group. This
is one reason the data-splitting method was later corrected to follow the paper
more closely. The complete table is in
`results/baseline/individual_baseline_results.csv`.

### Phase 5 - Independent MulCas Reconstruction

MulCas was rebuilt as a sequence of model decisions. A contract can be accepted
by an early model, passed to the next model, or sent to a voting stage. The
implementation preserved row order, recorded the decision thresholds, and
saved a trace showing what happened to each contract.

| Split | Precision | Recall | F1 | PR-AUC | Balanced acc. | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.5714 | 0.5000 | 0.5333 | 0.4160 | 0.7484 | 963 | 3 | 4 | 4 |
| Test | 0.6000 | 0.1084 | 0.1837 | 0.5222 | 0.5468 | 798 | 12 | 148 | 18 |

On the test group, six out of ten flagged contracts were actually Ponzi
(precision 0.6000), but the model found only about one in ten real Ponzi
contracts (recall 0.1084). This was an independent reconstruction, not an exact
replication of the paper.

### Phase 5.5 - Paper-Aligned Correction and Frozen Baseline

Phase 5.5 kept all earlier results and introduced a new split designed to match
the paper's description more closely:

- train through the 200th chronological Ponzi contract;
- validate after that through the 250th Ponzi contract;
- test after the 250th through the 314th Ponzi contract;
- assign benign contracts using the same creation-block boundaries;
- keep whole creation blocks together and never shuffle.

| Split | Rows | Ponzi | Benign | Benign:Ponzi | Block range |
| --- | ---: | ---: | ---: | ---: | --- |
| Train | 5,839 | 200 | 5,639 | 28.20:1 | 49,880-5,021,809 |
| Validation | 150 | 50 | 100 | 2.00:1 | 5,021,917-5,038,334 |
| Test | 509 | 64 | 445 | 6.95:1 | 5,038,337-7,136,486 |

The paper reports 5,990 rows through P4 and 508 in P5. This reconstruction has
5,989 and 509, a difference of one row. The difference was kept and documented
instead of manually moving a row just to match the paper's total.

Paper-aligned features preserve the old 70,867-dimensional N-grams as an
alternative while adding document-frequency-filtered N-grams and IDF-weighted
Word2Vec. The creator feature is appended to every static view.

| Paper-aligned representation | Final dimension including creator |
| --- | ---: |
| Term count | 143 |
| TF-IDF | 143 |
| Filtered 1-3-gram count | 44,672 |
| Filtered 1-3-gram TF-IDF | 44,672 |
| Mean Word2Vec | 101 |
| IDF-weighted Word2Vec | 101 |

Individual selected model results on the final test were:

| Model | Selected representation | Validation F1 | Test P | Test R | Test F1 | Test PR-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | Term count | 0.8519 | 0.9310 | 0.4219 | 0.5806 | 0.6365 |
| SVM | Filtered N-gram TF-IDF | 0.8519 | 0.8824 | 0.2344 | 0.3704 | 0.6717 |
| XGBoost | Filtered N-gram TF-IDF | 0.8519 | 0.8333 | 0.2344 | 0.3659 | 0.5943 |

The frozen reference MulCas sequence is:

1. Cascade viewer 1: XGBoost with opcode term count, confidence interval
   thresholds 0.05 and 0.95.
2. Cascade viewer 2: XGBoost with filtered N-gram count, thresholds 0.05 and
   0.51.
3. Voting stage: filtered N-gram TF-IDF XGBoost and opcode TF-IDF Ridge, with
   voting threshold `lambda = 0`.
4. The chronological creator feature is appended to every view.

Frozen MulCas results:

| Split | Accuracy | Balanced acc. | Precision | Recall | F1 | PR-AUC | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.8933 | 0.9000 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 88 | 12 | 4 | 46 |
| Final test | 0.9253 | 0.7232 | 0.9063 | 0.4531 | 0.6042 | 0.6624 | 442 | 3 | 35 | 29 |

Comparison with the publication:

| Result source | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Published paper, Table 5 | 0.9510 | 0.6740 | 0.7890 |
| Original 70/15/15 reconstruction | 0.6000 | 0.1084 | 0.1837 |
| Corrected Phase 5.5 reconstruction | 0.9063 | 0.4531 | 0.6042 |
| Phase 5.5 minus published | -0.0448 | -0.2209 | -0.1848 |

The corrected split improved the reconstruction substantially, but it still did
not reproduce the paper's recall or F1. The reconstructed final-test precision
was 0.9063, recall was 0.4531, and F1 was 0.6042. The paper reports different
values and even gives two different F1 values in different places. Because the
paper does not provide every preprocessing and model detail, this project must
be described as an independent reconstruction, not an exact replication.

Primary Phase 5.5 artifacts are in `results/baseline/paper_aligned/`, with the
human report at `docs/paper_aligned_baseline_report.md`.

## 5. Work Completed After the Frozen Baseline

### Shared Phase 6 Experimental Protocol

Phase 6 tested possible improvements without looking at the final test labels.
It used three expanding, forward-only checks inside the Phase 5.5 training
period. In each check, older contracts were used for training and the next
historical block was used for checking. Feature-building tools were retrained
inside each check so that future information could not leak backwards.

| Fold | Historical train | Later CV interval |
| --- | --- | --- |
| 1 | 104 rows, 50 Ponzi, blocks 49,880-1,287,252 | 305 rows, 50 Ponzi, blocks 1,287,353-2,000,473 |
| 2 | 409 rows, 100 Ponzi, blocks 49,880-2,000,473 | 5,141 rows, 50 Ponzi, blocks 2,008,140-4,963,650 |
| 3 | 5,550 rows, 150 Ponzi, blocks 49,880-4,963,650 | 289 rows, 50 Ponzi, blocks 4,964,082-5,021,809 |

Before running Phase 6, a candidate needed average precision of at least 0.80
across the three time-based checks. F1, recall, balanced accuracy, and simpler
settings were used only to break ties. The external validation group and final
test group did not choose the Phase 6 candidates.

### Phase 6A - Cost-Sensitive Learning

The models were told to pay more attention to the less common Ponzi class. No
variant passed the required time-based precision rule, so the reported choices
were diagnostic fallbacks rather than supported improvements. Their validation
classifications stayed similar, but their ranking scores were below the frozen
reference. **H6A was not supported.**

### Phase 6B - Training-Data Sampling

The training data was rebalanced in several ways: some benign examples were
removed, or Ponzi examples were repeated, at several positive-to-negative
ratios. Only the selected SVM candidate passed the time-based precision rule,
but its final validation behaviour was not better than the reference.

| Selected diagnostic | Validation P | R | F1 | PR-AUC | Main observation |
| --- | ---: | ---: | ---: | ---: | --- |
| Ridge, under 1:10 | 0.7931 | 0.9200 | 0.8519 | 0.7175 | Same classifications, worse ranking |
| SVM, over 1:10 | 0.7931 | 0.9200 | 0.8519 | 0.8074 | Same classifications, worse ranking |
| XGBoost, under 1:1 | 0.7302 | 0.9200 | 0.8142 | 0.8796 | More false positives, lower F1 |

**H6B was not supported.**

### Phase 6C - Decision-Threshold Optimization

Only the final yes/no cut-off inside each individual classifier was changed.
The MulCas cascade thresholds and voting rule were left unchanged. The aim was
to see whether a different cut-off produced a better precision-recall balance.

| Model | CV-selected threshold | Eligible | Selected validation P | R | F1 |
| --- | ---: | --- | ---: | ---: | ---: |
| Ridge | -0.9730 | No, diagnostic fallback | 0.5000 | 0.9800 | 0.6622 |
| SVM | 1.6765 | Yes | 0.0000 | 0.0000 | 0.0000 |
| XGBoost | 0.9936 | Yes | 0.0000 | 0.0000 | 0.0000 |

The eligible SVM and XGBoost settings were too strict and found no validation
Ponzi contracts. The Ridge fallback found more contracts but produced too many
false alarms, reducing precision and F1. **H6C was not supported.**

### Phase 6D1 - Structural Opcode Features

Seventy-four extra features were calculated directly from each contract's
opcode structure, including size, variety, instruction groups, stack patterns,
and repetition. Exact operand values were not used. No candidate passed the
time-based precision rule, and the extra features did not improve ranking.
**H6D1 was not supported.**

### Phase 6D2 - Normalized Operand Features

Thirty-seven additional features described the general shape of bytecode
operands, such as their size, frequency, and magnitude. Exact hexadecimal
values, selectors, and addresses were not exposed. The parser recorded malformed
or unsupported instructions and handled them consistently; affected contracts
were not silently removed. No candidate passed the time-based precision rule.
**H6D2 was not supported.**

| Selected diagnostic | Validation P | R | F1 | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Ridge, operand only | 0.6667 | 0.0800 | 0.1429 | 0.7632 |
| SVM, baseline + width group | 0.7931 | 0.9200 | 0.8519 | 0.8354 |
| XGBoost, baseline + width group | 0.7931 | 0.9200 | 0.8519 | 0.8681 |

No candidate met the CV precision floor. **H6D2 was not supported.**

### Phase 6E - Alternative Classifiers

Two other model types, Logistic Regression and LightGBM, were tested using the
existing Phase 5.5 feature views. Neither produced a candidate that passed the
time-based selection rule. **H6E was not supported.**

| Selected diagnostic | Validation P | R | F1 | PR-AUC | Balanced acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression, term count, C=0.1 | 0.7931 | 0.9200 | 0.8519 | 0.7635 | 0.9000 |
| LightGBM, term count | 0.8036 | 0.9000 | 0.8491 | 0.8386 | 0.8950 |

Logistic Regression matched classifications but ranked cases less well.
LightGBM gained a little precision while losing recall, F1, PR-AUC, and balanced
accuracy. **H6E was not supported.**

### Phase 6 Closure

**None of the tested Phase 6 changes reliably improved the reference model.**
This does not mean that improvement is impossible. Phase 6F was not run because
there was no supported improvement to combine with another change. The Phase
5.5 MulCas model therefore remained the reference model for Phase 7.

See `docs/phase6_summary.md` and
`results/experiments/phase6_summary.json` for the closure record.

### Phase 7A - Temporal Robustness and Model Aging

The full dataset was divided into six time periods. Each period contains whole
creation blocks and approximately 50 Ponzi contracts, except the final period,
which contains the remaining 64. The model was trained on earlier periods and
tested on the next period.

| Period | Rows | Ponzi | Benign | Block range |
| --- | ---: | ---: | ---: | --- |
| P0 | 104 | 50 | 54 | 49,880-1,287,252 |
| P1 | 305 | 50 | 255 | 1,287,353-2,000,473 |
| P2 | 5,141 | 50 | 5,091 | 2,008,140-4,963,650 |
| P3 | 289 | 50 | 239 | 4,964,082-5,021,809 |
| P4 | 150 | 50 | 100 | 5,021,917-5,038,334 |
| P5 | 509 | 64 | 445 | 5,038,337-7,136,486 |

For example, P2 was predicted using a model trained on P0 and P1. Then P3 was
predicted using P0 through P2. This simulates a model that is periodically
updated as new historical contracts become available.

| Predicted period | Precision | Recall | F1 | PR-AUC | Balanced acc. | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 0.0492 | 0.2600 | 0.0828 | 0.0462 | 0.6053 | 4,840 | 251 | 37 | 13 |
| P3 | 0.5385 | 0.1400 | 0.2222 | 0.4786 | 0.5574 | 233 | 6 | 43 | 7 |
| P4 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9000 | 88 | 12 | 4 | 46 |
| P5 | 0.8710 | 0.4219 | 0.5684 | 0.6674 | 0.7064 | 441 | 4 | 37 | 27 |

Published and reconstructed temporal precision/recall/F1 are:

| Period | Published P / R / F1 | Reconstruction P / R / F1 |
| --- | --- | --- |
| P2 | 0.8800 / 0.3800 / 0.5300 | 0.0492 / 0.2600 / 0.0828 |
| P3 | 0.9600 / 0.3200 / 0.4800 | 0.5385 / 0.1400 / 0.2222 |
| P4 | 0.8100 / 0.9400 / 0.8700 | 0.7931 / 0.9200 / 0.8519 |
| P5 | 0.9500 / 0.6700 / 0.7900 | 0.8710 / 0.4219 / 0.5684 |

Routing through the frozen system also changes by period:

| Period | Accepted by viewer 1 | Accepted by viewer 2 | Reached voting |
| --- | ---: | ---: | ---: |
| P2 | 4,859 | 263 | 19 |
| P3 | 271 | 13 | 5 |
| P4 | 134 | 15 | 1 |
| P5 | 454 | 29 | 26 |

Performance changes sharply over time. In P2 the model raised many false alarms.
In P3 it found very few Ponzi contracts. P4 was unusually strong. In P5 it was
usually correct when it raised an alarm, but it still missed more than half of
the Ponzi contracts. These results show changing performance, but they do not
by themselves prove a specific cause such as concept drift. They do show why a
single train/test split is not enough to claim long-term reliability.

### Phase 7B - Frozen MulCas Component Ablation

One frozen MulCas part was removed at a time. No threshold or model was changed
afterward. The table shows whether each part helped in each time period.

| Ablation | P2 | P3 | P4 | P5 |
| --- | ---: | ---: | ---: | ---: |
| Full frozen MulCas | 0.0828 | 0.2222 | 0.8519 | 0.5684 |
| Remove first cascade viewer | 0.0755 | 0.1786 | 0.8519 | 0.3704 |
| Remove second cascade viewer | 0.0984 | 0.1455 | 0.8519 | 0.4944 |
| Remove voting stage | 0.0823 | 0.2222 | 0.8519 | 0.5532 |
| First viewer only | 0.0984 | 0.0000 | 0.8462 | 0.4944 |
| Second viewer only | 0.0755 | 0.0000 | 0.8381 | 0.3704 |

The first viewer mattered most in P5. The second viewer helped in some periods
and hurt in others. Voting made a small contribution. A single viewer was not
reliable across all periods, so none of these reduced systems replaces the
frozen MulCas model.

### Phase 7C - Chronological Creator Feature Ablation

The creator-history number was removed from every feature view. Everything else
stayed the same: models, thresholds, routing, and time periods. This isolates
the effect of that one feature.

| Period | With creator F1 | Without creator F1 | Delta | Recall delta | FP delta | FN delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 0.0828 | 0.1048 | +0.0220 | -0.0400 | -102 | +2 |
| P3 | 0.2222 | 0.3099 | +0.0876 | +0.0800 | +4 | -4 |
| P4 | 0.8519 | 0.8519 | +0.0000 | +0.0000 | 0 | 0 |
| P5 | 0.5684 | 0.5979 | +0.0295 | +0.0312 | 0 | -2 |

Creator-feature prevalence was sparse:

| Period | Previously observed creator | Creator flag = 1 | Ponzi flag = 1 | Benign flag = 1 |
| --- | ---: | ---: | ---: | ---: |
| P2 | 148 | 1 | 0 | 1 |
| P3 | 49 | 2 | 1 | 1 |
| P4 | 20 | 6 | 4 | 2 |
| P5 | 85 | 11 | 4 | 7 |

The creator feature did not improve recall consistently. P4 predictions were
exactly the same with or without it. Removing it improved F1 in P3 and P5. In
P2 it removed 102 false alarms but also lost two correctly found Ponzi
contracts. In P5, 36 of 37 missed Ponzi contracts came from creators with no
earlier Ponzi history. The feature is therefore a small historical signal; it
does not show that a creator's identity causes a contract to be a Ponzi.

## 6. Overall Findings So Far

1. **Protocol design matters.** The paper-aligned split and representations
   improved test F1 from 0.1837 to 0.6042 compared with the initial independent
   reconstruction.
2. **The paper result was not exactly reproduced.** Phase 5.5 approached its
   precision but remained substantially lower in recall and F1.
3. **Phase 6 did not identify a reliable improvement.** Cost weighting,
   sampling, thresholds, structural features, operand features, Logistic
   Regression, and LightGBM all failed the predefined temporal evidence rule.
4. **Performance is not temporally stable.** F1 ranges from 0.0828 in P2 to
   0.8519 in P4, then falls to 0.5684 in P5.
5. **MulCas components are complementary but period-dependent.** The first
   viewer matters strongly in P5; the second viewer and voting stage have more
   modest or mixed effects.
6. **Creator history is not uniformly beneficial.** Its effect changes by
   period, and most late false negatives involve creators with no prior Ponzi
   history.
7. **The correct reference remains Phase 5.5.** Phase 6 diagnostics and Phase 7
   ablations are analyses, not replacement production models.

## 7. Repository Map

- `configs/`: frozen baseline and experiment YAML files.
- `data/raw/`: immutable local source CSV; ignored by Git.
- `data/processed/`: intentionally unused for copied datasets or matrices.
- `docs/`: reports, research notes, protocol, and this progress report.
- `notebooks/`: thin presentation notebooks `00` through `07`; notebook `08`
  is reserved for future final evaluation.
- `src/data/`: loading, validation, auditing, and temporal splitting.
- `src/features/`: baseline, paper-aligned, structural, and operand extraction.
- `src/models/`: baseline estimators, evaluation, and MulCas implementation.
- `src/experiments/`: Phase 6 and Phase 7 experiment runners.
- `src/evaluation/`: metrics and temporal evaluation helpers.
- `tests/`: unit, leakage, determinism, frozen-manifest, and experiment tests.
- `results/baseline/`: compact baseline outputs.
- `results/experiments/`: Phase 6 and Phase 7 machine-readable outputs.
- `results/artifacts/`: ignored fitted models/transformers and large artifacts.

`data/processed/` can remain empty because the project deliberately stores
split membership and compact summaries instead of duplicating the raw CSV or
committing large feature matrices. Trained objects belong under
`results/artifacts/`, not `data/processed/`.

## 8. Local Setup on Windows PowerShell

Run all commands from the repository root.

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm the interpreter and imports:

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import numpy, pandas, sklearn, xgboost, lightgbm, gensim; print('Imports OK')"
```

`sys.executable` should point to `.venv\Scripts\python.exe`.

Verify the raw dataset without editing it:

```powershell
Test-Path -LiteralPath data\raw\PonziDataset_20221114.csv
Get-FileHash -Algorithm SHA256 data\raw\PonziDataset_20221114.csv
```

The expected SHA-256 is the value recorded in Section 3.

## 9. How to Test the Current Project

Run the complete suite:

```powershell
python -m pytest -q
```

At the time of this report, the suite contains **100 passing tests**. A pytest
cache permission warning can occur on Windows without representing test
failure; use the final pass/fail summary.

Useful focused checks:

```powershell
python -m pytest tests/test_data_validation.py tests/test_temporal_split.py -q
python -m pytest tests/test_feature_leakage.py tests/test_paper_aligned_protocol.py -q
python -m pytest tests/test_mulcas.py tests/test_evaluation_metrics.py -q
python -m pytest tests/test_phase6a.py tests/test_phase6b.py tests/test_phase6c.py -q
python -m pytest tests/test_phase6d1.py tests/test_phase6d2.py tests/test_phase6e.py -q
python -m pytest tests/test_phase7a.py tests/test_phase7b.py tests/test_phase7c.py -q
```

Check that the raw data and frozen artifacts were not accidentally changed:

```powershell
git diff -- data/raw/PonziDataset_20221114.csv
git status --short
```

The raw-data diff should be empty. Review any baseline or frozen-manifest
changes before accepting regenerated outputs.

## 10. How to Reproduce the Pipeline

### Baseline through Phase 5.5

```powershell
python -m src.data.audit
python -m src.data.split --config configs/data_split.yaml
python -m src.features.reproduce --config configs/feature_reproduction.yaml
python -m src.models.run_baselines --config configs/baseline_models.yaml
python -m src.models.run_mulcas --config configs/mulcas_reproduction.yaml

python -m src.data.paper_aligned_split --config configs/paper_aligned_split.yaml
python -m src.features.paper_aligned --config configs/paper_aligned_features.yaml
python -m src.models.run_baselines --config configs/paper_aligned_baseline_models.yaml
python -m src.models.run_mulcas --config configs/paper_aligned_mulcas.yaml
```

These commands retrain models and regenerate baseline outputs. Do not run them
merely to inspect existing results; open the saved JSON/CSV/report files instead.

### Archived Phase 6 Experiments

Each command can be run independently after Phase 5.5 artifacts exist:

```powershell
python -m src.experiments.phase6a --config configs/experiments/phase6a/cost_sensitive.yaml
python -m src.experiments.phase6b --config configs/experiments/phase6b/training_sampling.yaml
python -m src.experiments.phase6c --config configs/experiments/phase6c/threshold_optimization.yaml
python -m src.experiments.phase6d1 --config configs/experiments/phase6d1/structural_features.yaml
python -m src.experiments.phase6d2 --config configs/experiments/phase6d2/operand_features.yaml
python -m src.experiments.phase6e --config configs/experiments/phase6e/alternative_classifiers.yaml
```

These are archived negative experiments. Rerunning them is for reproducibility,
not an invitation to select candidates after seeing validation results.

### Phase 7 Analyses

Run in order because 7B and 7C reuse frozen prior outputs:

```powershell
python -m src.experiments.phase7a --config configs/phase7/temporal_robustness.yaml
python -m src.experiments.phase7b --config configs/phase7/view_ablation.yaml
python -m src.experiments.phase7c --config configs/phase7/creator_ablation.yaml
```

### Notebooks

Start Jupyter from the activated environment:

```powershell
python -m jupyter notebook
```

Notebooks `00_environment_check.ipynb` through `07_ablation_study.ipynb` are
presentation layers that read generated artifacts. Core logic belongs under
`src/`. Notebook `08_final_evaluation.ipynb` is not yet an executed final study.

Non-interactive execution example:

```powershell
New-Item -ItemType Directory -Force results\temp | Out-Null
python -m jupyter nbconvert --execute --to notebook `
  --output-dir results\temp `
  --output ablation_study_executed.ipynb `
  notebooks\07_ablation_study.ipynb
```

## 11. What Researchers Should Do

- Keep the raw CSV byte-for-byte unchanged and verify its SHA-256.
- Activate `.venv` and run commands from the repository root.
- Treat Phase 5.5 as the frozen reference system.
- Use chronological, forward-only evaluation for future work.
- Fit every learned transform only on the applicable historical training rows.
- Preserve creator-history semantics: only strictly earlier labels are known.
- Keep validation, final test, and published metrics clearly labeled.
- Add new work under new config/result namespaces and record hashes and seeds.
- Run focused tests while developing and all tests before reporting results.
- Store compact metrics and traces in Git; keep large fitted artifacts under
  ignored `results/artifacts/` paths.
- Interpret temporal changes descriptively unless stronger drift evidence is
  established by an appropriate statistical analysis.

## 12. What Researchers Should Not Do

- Do not edit, sort, deduplicate, clean, or resave the raw CSV.
- Do not shuffle before chronological partition assignment.
- Do not fit vocabularies, IDF, embeddings, scalers, or models on future rows.
- Do not use validation or test labels to tune after viewing their results.
- Do not evaluate the final test repeatedly while developing improvements.
- Do not overwrite Phase 5.5 or archived Phase 6/7 outputs.
- Do not combine failed Phase 6 interventions post hoc and call the result a
  prespecified experiment.
- Do not treat a diagnostic fallback as an eligible winner.
- Do not claim the reconstructed values are the paper's published values.
- Do not infer that creator identity causes fraud from a predictive flag.
- Do not claim concept drift is proven solely from changing period metrics.
- Do not commit `.venv`, raw/local datasets, secrets, caches, large matrices, or
  fitted model files.
- Do not place reusable research logic only in notebooks.

## 13. Key Result Locations

- Dataset audit: `results/baseline/dataset_audit.json`
- Initial split: `results/baseline/temporal_split_metadata.json`
- Initial features: `results/baseline/feature_view_summary.json`
- Initial model results: `results/baseline/individual_baseline_results.csv`
- Initial MulCas: `results/baseline/mulcas_reproduction_results.json`
- Phase 5.5 baseline: `results/baseline/paper_aligned/mulcas_results.json`
- Phase 5.5 report: `docs/paper_aligned_baseline_report.md`
- Phase 6 closure: `results/experiments/phase6_summary.json`
- Phase 7A: `results/experiments/phase7a/phase7a_results.json`
- Phase 7B: `results/experiments/phase7b/phase7b_results.json`
- Phase 7C: `results/experiments/phase7c/phase7c_results.json`
- Experiment chronology: `docs/experiment_log.md`

The JSON and CSV files are the machine-readable source of truth. Markdown
reports summarize them for explanation and presentation.

