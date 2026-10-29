# Phase 9B Prospective Evaluation Protocol

## Purpose

Phase 9B freezes the evaluation rules before Phase 9C model development. It does not train, calibrate, select, or score a detector.

## RQ7

To what extent can deployment-available historical state, code-family similarity, and calibrated cascade routing improve forward temporal Ponzi detection under a leakage-safe prospective protocol, and does the final frozen system generalize to a sealed external later-time benchmark?

## Development Boundary

Phase 9 development uses Track A only. Forward predictions are `P0+P1 -> P2`, `P0-P2 -> P3`, and `P0-P3 -> P4`. P5 is previously observed and is excluded from model selection. Track B labels remain inaccessible throughout Phases 9C-9F.

## Metrics And Selection

Primary metrics are pooled prequential PR-AUC, pooled prequential F1, and recall at the frozen MulCas pooled FPR ceiling `0.049539594843`. Secondary metrics include balanced accuracy, ROC-AUC, per-period precision/recall/F1, and prevalence-standardized precision at `0.048322560788`.

A candidate advances only when pooled F1 is strictly higher, pooled PR-AUC degrades by no more than `0.01`, FPR does not exceed the frozen MulCas ceiling, at least two of three periods have nonnegative F1 change, no period loses more than `0.02` F1, and paired uncertainty is reported. Paired contract-level bootstrap uses 2,000 replicates and seed 42.

## Planned Interventions

Only mechanism-driven Phases 9C-9F are preregistered: deployment-available creator/family state, cascade calibration and safer routing, selector/semantic static features, and combination only of independently supported components. GIN, Transformer, creation-code model, unrestricted threshold search, and arbitrary model-zoo searches remain closed.

## External Evaluation

Track B status at freeze: **sealed_available**. A final Phase 9F system must be frozen before one external evaluation can access sealed labels. No tuning is permitted afterward. Strict and provisional subsets must remain distinct.

## Integrity Statement

This protocol was frozen without loading Track B detector predictions or performance. No detector was trained or scored, and Phase 9C was not started.
