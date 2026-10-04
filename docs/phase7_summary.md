# Phase 7 Research Closure

## Status and Scope

Phase 7 is complete. It evaluated the frozen Phase 5.5 paper-aligned MulCas
reconstruction through temporal robustness analysis, component and creator
ablation, systematic error analysis, and exact historical-clone sensitivity.
Phase 7 did not optimize the system, introduce a replacement model, tune a
threshold, or enable any Phase 6 intervention.

The analyses reused the same chronological P0-P5 definitions and the frozen
Phase 7A predictions wherever possible. All claims below are descriptive.

## Phase 7A: Temporal Robustness

The frozen system was trained on expanding historical periods and evaluated on
the immediately following period.

| Predict | Historical training | Precision | Recall | F1 | PR-AUC | Balanced accuracy | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | P0 + P1 | 0.0492 | 0.2600 | 0.0828 | 0.0462 | 0.6053 | 4,840 | 251 | 37 | 13 |
| P3 | P0 + P1 + P2 | 0.5385 | 0.1400 | 0.2222 | 0.4786 | 0.5574 | 233 | 6 | 43 | 7 |
| P4 | P0 + P1 + P2 + P3 | 0.7931 | 0.9200 | 0.8519 | 0.8706 | 0.9000 | 88 | 12 | 4 | 46 |
| P5 | P0 + P1 + P2 + P3 + P4 | 0.8710 | 0.4219 | 0.5684 | 0.6674 | 0.7064 | 441 | 4 | 37 | 27 |

Performance is strongly non-monotonic. P2 produces many false positives, P3
has very low recall, P4 is substantially stronger, and P5 retains high
precision while missing 37 of 64 Ponzi contracts. Accuracy alone is misleading
in P2 because that period contains 5,091 benign contracts and only 50 Ponzi
contracts.

The reconstruction is below the directly comparable published values in every
period, although P4 is close.

| Period | Published P / R / F1 | Reconstruction P / R / F1 | F1 difference |
| --- | --- | --- | ---: |
| P2 | 0.8800 / 0.3800 / 0.5300 | 0.0492 / 0.2600 / 0.0828 | -0.4472 |
| P3 | 0.9600 / 0.3200 / 0.4800 | 0.5385 / 0.1400 / 0.2222 | -0.2578 |
| P4 | 0.8100 / 0.9400 / 0.8700 | 0.7931 / 0.9200 / 0.8519 | -0.0181 |
| P5 | 0.9500 / 0.6700 / 0.7900 | 0.8710 / 0.4219 / 0.5684 | -0.2216 |

For RQ5, Phase 7A shows that effectiveness on later contracts is not stable
across periods. Because the prediction period and amount of historical training
both change, this result describes operational temporal performance rather than
isolating a single cause.

## Phase 7B: MulCas Component Ablation

The first cascade viewer, second cascade viewer, and voting stage were removed
one at a time without retuning.

| Ablation | P2 F1 delta | P3 F1 delta | P4 F1 delta | P5 F1 delta | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Remove first cascade viewer | -0.0073 | -0.0437 | 0.0000 | -0.1981 | Strongest positive contribution in P5 |
| Remove second cascade viewer | +0.0156 | -0.0768 | 0.0000 | -0.0740 | Mixed, period-dependent contribution |
| Remove voting stage | -0.0005 | 0.0000 | 0.0000 | -0.0152 | Small nonnegative contribution |

Removing the first viewer reduces P5 recall from 0.4219 to 0.2344. The second
viewer slightly hurts P2 F1 but helps P3 and P5. Removing voting changes only
four P2 predictions, no P3/P4 predictions, and one P5 prediction. Standalone
viewers fail badly in some periods, including zero recall for both standalone
viewers in P3.

For RQ4, the viewers are complementary, but their contribution is not stable
over time. Routing volume alone does not explain performance because viewer 1
handles most contracts in every period while outcomes vary sharply.

## Phase 7C: Creator-Feature Ablation

The chronological creator feature was removed from every view without adding a
replacement column. Delta means **without creator minus with creator**.

| Period | Creator flag = 1 | Delta precision | Delta recall | Delta F1 | Delta PR-AUC | Delta FP | Delta FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 1/5,141 | +0.0195 | -0.0400 | +0.0220 | +0.0261 | -102 | +2 |
| P3 | 2/289 | -0.0147 | +0.0800 | +0.0876 | +0.1453 | +4 | -4 |
| P4 | 6/150 | 0.0000 | 0.0000 | 0.0000 | -0.0087 | 0 | 0 |
| P5 | 11/509 | +0.0078 | +0.0312 | +0.0295 | +0.0165 | 0 | -2 |

Creator history is sparse and not uniformly beneficial. P4 classifications are
unchanged, and removing the creator feature recovers two P5 positives without
adding false positives. In P2, retaining the feature finds two additional
positives but is associated with 102 additional false positives through learned
model interactions. The creator flag is a historical predictive signal, not a
causal statement about creator behavior.

## Phase 7D: Systematic Error Analysis

| Period | False positives | False negatives | Errors from unseen creators | Dominant error route |
| --- | ---: | ---: | ---: | --- |
| P2 | 251 | 37 | 98.3% | 250 second-viewer false positives |
| P3 | 6 | 43 | 98.0% | 41 first-viewer false negatives |
| P4 | 12 | 4 | 87.5% | 11 first-viewer errors |
| P5 | 4 | 37 | 92.7% | 29 first-viewer errors |

Most errors are accepted by a cascade viewer before voting. Only 1.7%-19.5%
of period errors reach voting. Error characteristics also vary by period: there
is no single opcode-length, entropy, structural, or operand profile that
separates errors consistently.

Among the 37 P5 false negatives:

- 36 have unseen creators and 36 lack the historical Ponzi-creator flag;
- all 37 lack an exact historical opcode or runtime-bytecode clone;
- 28 are confidently rejected by viewer 1 and one by viewer 2;
- eight remain unresolved until voting; and
- only three are near a configured boundary under the predefined margin.

These patterns are evidence consistent with temporal distribution variation.
They do not prove concept drift and do not identify a causal mechanism.

## Phase 7E: Clone and Novelty Sensitivity

Exact clone status uses normalized opcode sequences, runtime bytecode, and
creation code from strictly earlier training periods only.

| Period | Any clone prevalence | Cloned Ponzi | Recall cloned Ponzi | Recall novel Ponzi |
| --- | ---: | ---: | ---: | ---: |
| P2 | 29/5,141 (0.6%) | 0/50 | N/A | 26.0% |
| P3 | 1/289 (0.3%) | 0/50 | N/A | 14.0% |
| P4 | 15/150 (10.0%) | 14/50 | 100.0% | 88.9% |
| P5 | 8/509 (1.6%) | 1/64 | 100.0% | 41.3% |

The P5 creator/clone interaction is concentrated in one group:

| P5 Ponzi group | Contracts | TP | FN | Recall |
| --- | ---: | ---: | ---: | ---: |
| Seen creator + historical clone | 1 | 1 | 0 | 100.0% |
| Seen creator + novel code | 3 | 2 | 1 | 66.7% |
| Unseen creator + novel code | 60 | 24 | 36 | 40.0% |

All 37 P5 false negatives are historically novel. Among 63 novel P5 Ponzi
contracts, 29 are confidently misclassified before voting and eight are
misclassified after reaching voting.

Historical code reuse is associated with easier detection where cloned Ponzi
contracts occur. However, clone subgroups are very small, P2/P3 contain no
cloned positives, and P4 still achieves 88.9% recall on novel positives. Clone
prevalence therefore does not fully explain temporal performance variation.
Novel code is an observed subgroup characteristic, not a demonstrated cause of
misclassification.

## Consolidated Evidence

| Research question | Evidence | Conservative answer |
| --- | --- | --- |
| RQ4: Which feature groups and MulCas components contribute most strongly? | Phase 7B finds the first viewer especially important in P5, a mixed second-viewer effect, and a small voting contribution. Phase 7C finds sparse, period-dependent creator effects. Phase 7D shows the viewer producing errors also changes by period. | Contributions are complementary and period-dependent. No component or creator feature is uniformly beneficial across P2-P5. |
| RQ5: Does reconstructed MulCas retain effectiveness on later contracts? | F1 varies from 0.0828 in P2 and 0.2222 in P3 to 0.8519 in P4 and 0.5684 in P5. P5 retains high precision but has 0.4219 recall. Later errors concentrate among unseen creators and novel code. | The model retains useful performance in some later periods but is not temporally stable. The evidence is consistent with distribution variation, not proof of concept drift. |
| RQ4 and RQ5 interaction | P5 depends strongly on viewer 1, yet 28 P5 false negatives are confidently rejected there. Creator history and exact clones are sparse. P4 performs strongly even on novel positives. | Architecture, historical identity signals, code reuse, period composition, and learned representations interact; no single examined factor fully explains the temporal results. |

## Overall Conclusion

The reconstructed MulCas model exhibits substantial temporal performance
variation. Component and creator-feature contributions are period-dependent.
Later-period errors are concentrated among contracts with unseen creators and
historically novel code, and many are confidently misclassified. Exact
historical code reuse is associated with easier detection where cloned Ponzi
contracts occur, but clone prevalence does not fully explain temporal
performance variation.

Phase 7 does **not** prove concept drift. It also does not show that novel code
causes misclassification. No model was selected or modified from these
analyses; the frozen Phase 5.5 reconstruction remains the reference system.

## Archived Outputs

- Phase 7A: [`results/experiments/phase7a/phase7a_results.json`](../results/experiments/phase7a/phase7a_results.json)
- Phase 7B: [`results/experiments/phase7b/phase7b_results.json`](../results/experiments/phase7b/phase7b_results.json)
- Phase 7C: [`results/experiments/phase7c/phase7c_results.json`](../results/experiments/phase7c/phase7c_results.json)
- Phase 7D: [`results/experiments/phase7d/phase7d_results.json`](../results/experiments/phase7d/phase7d_results.json)
- Phase 7E: [`results/experiments/phase7e/phase7e_results.json`](../results/experiments/phase7e/phase7e_results.json)

The machine-readable closure record is
[`results/experiments/phase7_summary.json`](../results/experiments/phase7_summary.json).
