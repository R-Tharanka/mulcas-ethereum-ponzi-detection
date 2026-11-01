# Phase 8: Static Extensions and Track B

This package contains the largest research extension:

- Phase 8R0 extension governance and feasibility;
- Phase 8R1 recency-aware learning;
- Phase 8R2 runtime CFG construction and GIN evaluation;
- Phase 8R3 MulCas/GIN complementarity and score fusion;
- Phase 8R4 creation-code representations and classification;
- Phase 8R5 self-supervised opcode-sequence learning; and
- Phase 8R6 external-source enrichment and sealed Track B construction.

No tested Phase 8 intervention replaced frozen Phase 5.5 MulCas.

Important locations:

- `configs/extension/`: Phase 8 configs and predecessor manifests;
- `src/extension/`: graph, creation-code, sequence, provider, and sealing logic;
- `tests/test_phase8*.py`: phase and integrity tests;
- `results/extension/`: Phase 8 outputs and closure evidence;
- `results/extension/phase8r6d/enriched_contracts_final.csv`: complete
  provider-enriched external table; and
- `docs/phase8_summary.md`: formal Phase 8 summary.
