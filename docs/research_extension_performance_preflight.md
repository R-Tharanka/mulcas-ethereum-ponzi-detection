# Extension Performance Preflight Policy

This addendum extends the frozen research extension protocol without modifying
its Phase 8R0 archival copy. It applies project-wide to future extension work.

Before any future experiment expected to involve repeated model training:

1. Run one representative benchmark job.
2. Measure its elapsed runtime.
3. Record the CPU/GPU device, PyTorch version, CUDA availability, GPU model,
   and available RAM/VRAM where the environment exposes them.
4. Estimate total runtime for the complete proposed experiment from the
   representative benchmark and declared run count.
5. Do not launch the full experiment automatically when the estimated runtime
   exceeds 30 minutes. Obtain explicit user approval first.

This rule changes no completed experimental result or conclusion. The original
[`research_extension_protocol.md`](research_extension_protocol.md) remains
hash-frozen as a Phase 8R0 artifact.
