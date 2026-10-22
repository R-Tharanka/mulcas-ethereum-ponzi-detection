# Track B Ponzi Smart-Contract Deployment Timeline Report

## 1. Report purpose

This report documents the deployment timeline of Ponzi-labelled smart contracts
in the newly established external benchmark, **Track B**. It describes when the
contracts were created on Ethereum and summarizes how the Ponzi deployments are
distributed over time.

## 2. Dataset overview

Track B is a new, separate external benchmark containing **292 unique smart
contracts**:

| Class | Contracts | Share of Track B |
| --- | ---: | ---: |
| Ponzi | 112 | 38.36% |
| Benign | 180 | 61.64% |
| **Total** | **292** | **100.00%** |

Track B is not an extension of the original Track A CSV. The original Track A
dataset still contains **6,498 contracts**, including 314 Ponzi contracts, and
remains byte-for-byte unchanged.

Track B was established as a strict external benchmark after provider-backed
enrichment and checks for chronology, contract identity, label conflicts, and
address, runtime-code, opcode, and creation-code overlap. All 292 selected
contracts satisfied the strict Track B criteria. Track B contains 287 unique
runtime-code and opcode identities.

## 3. Method used to construct the timeline

The timeline was calculated by joining:

1. `data/external/track_b/metadata.csv`, which contains each contract address,
   creation block, and Ethereum creation timestamp; and
2. `data/external/track_b_sealed/labels.csv`, which assigns the binary label
   `1` for Ponzi and `0` for benign.

Only the 112 rows labelled as Ponzi were included in the timeline. Monthly and
annual counts were calculated from their stored UTC creation timestamps.

## 4. Overall Ponzi deployment period

| Measure | Result |
| --- | --- |
| Ponzi contracts | 112 |
| Earliest Ponzi deployment | 19 February 2019, 13:59:52 UTC |
| Earliest Ponzi deployment block | 7,240,598 |
| Latest Ponzi deployment | 21 June 2020, 20:22:15 UTC |
| Latest Ponzi deployment block | 10,311,201 |
| Observed deployment span | Approximately 488 days |

The complete Track B benchmark, including both classes, starts slightly earlier:
its overall range is 29 January 2019 through 21 June 2020. The first
Ponzi-labelled contract in Track B was deployed on 19 February 2019.

## 5. Annual distribution

| Year | Ponzi contracts | Share of Track B Ponzi contracts |
| --- | ---: | ---: |
| 2019 | 36 | 32.14% |
| 2020 | 76 | 67.86% |
| **Total** | **112** | **100.00%** |

More than two-thirds of the Track B Ponzi contracts were deployed during the
first six months of 2020.

## 6. Monthly deployment timeline

| Month | Ponzi contracts | Share of Track B Ponzi contracts |
| --- | ---: | ---: |
| February 2019 | 3 | 2.68% |
| March 2019 | 2 | 1.79% |
| April 2019 | 5 | 4.46% |
| May 2019 | 0 | 0.00% |
| June 2019 | 4 | 3.57% |
| July 2019 | 3 | 2.68% |
| August 2019 | 4 | 3.57% |
| September 2019 | 3 | 2.68% |
| October 2019 | 2 | 1.79% |
| November 2019 | 8 | 7.14% |
| December 2019 | 2 | 1.79% |
| January 2020 | 13 | 11.61% |
| February 2020 | 3 | 2.68% |
| March 2020 | 11 | 9.82% |
| April 2020 | 4 | 3.57% |
| May 2020 | 23 | 20.54% |
| June 2020 | 22 | 19.64% |
| **Total** | **112** | **100.00%** |

## 7. Quarterly summary

| Quarter | Ponzi contracts | Share of Track B Ponzi contracts |
| --- | ---: | ---: |
| 2019 Q1 | 5 | 4.46% |
| 2019 Q2 | 9 | 8.04% |
| 2019 Q3 | 10 | 8.93% |
| 2019 Q4 | 12 | 10.71% |
| 2020 Q1 | 27 | 24.11% |
| 2020 Q2 | 49 | 43.75% |
| **Total** | **112** | **100.00%** |

The strongest concentration occurs in **2020 Q2**, which contains 49 of the 112
Ponzi contracts. May and June 2020 alone contain 45 contracts, or 40.18% of all
Ponzi-labelled contracts in Track B.

## 8. Interpretation

The distribution shows that the Track B Ponzi cohort becomes increasingly
concentrated toward the end of the benchmark period. There are 36 Ponzi
deployments in 2019 and 76 during January-June 2020. May 2020 has the largest
monthly count, with 23 contracts, followed by June 2020, with 22.

This makes Track B useful as a later external benchmark: it tests a detector on
contracts deployed after the periods used to construct the original benchmark,
while keeping Track A unchanged. However, this concentration also means that
Track B should not be interpreted as a uniformly sampled history of every
Ethereum Ponzi contract deployed during 2019-2020.

## 9. Scope and limitations

The dates in this report are **Ethereum smart-contract creation timestamps**.
They establish when each contract was deployed on-chain. They do not establish:

- when a scheme began accepting investments;
- when promotional or off-chain activity began;
- when a scheme stopped operating;
- when investigators or the public identified it as fraudulent; or
- the duration or financial impact of the scheme.

Track B provides a binary classification label only: **Ponzi** or **benign**. It
does not contain a verified subtype taxonomy for matrix, chain-referral, fixed
return, pyramid, ROI, or other Ponzi-scheme structures. Therefore, this report
describes the timeline of Ponzi-labelled contracts, not a timeline divided by
Ponzi scheme type.

The observed monthly counts describe this curated benchmark and should not be
treated as population-level estimates of Ethereum Ponzi activity.

## 10. Evidence sources

- `results/extension/phase8r6d/phase8r6d_results.json`: final Track B counts,
  class balance, temporal range, readiness decision, and integrity summary.
- `results/extension/phase8r6d/temporal_classification.csv`: row-level temporal
  classification for the enriched candidate contracts.
- `data/external/track_b/metadata.csv`: sealed Track B contract metadata,
  including creation blocks and UTC timestamps.
- `data/external/track_b_sealed/labels.csv`: sealed binary labels used to select
  the 112 Ponzi contracts.
- `docs/COMPLETE_PROJECT_PROGRESS_REPORT.md`: Phase 8R6A-8R6D research process,
  enrichment funnel, and Track B establishment summary.

## 11. Conclusion

Track B is a separate, strict external benchmark of 292 contracts, comprising
112 Ponzi and 180 benign contracts. Its Ponzi-labelled contracts were deployed
between 19 February 2019 and 21 June 2020. The timeline is weighted toward 2020,
particularly May and June, but it represents contract deployment dates within a
curated research benchmark rather than the complete operational history of
Ethereum Ponzi schemes.
