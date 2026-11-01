# Cross-Phase Synthesis and Support

This package contains files that describe or support several phases rather than
one experiment:

- complete and current progress reports;
- experiment logs and research protocols;
- run and test guidance;
- shared dependency files and test configuration;
- repository-level README and licensing material; and
- final synthesis placeholders.

Important locations:

- `docs/COMPLETE_PROJECT_PROGRESS_REPORT.md`: principal historical report;
- `docs/FULL_CURRENT_RESEARCH_PROGRESS_REPORT.md`: expanded progress report;
- `docs/experiment_log.md`: chronological execution record;
- `docs/RUN_AND_TEST_GUIDE_pre_random_flow_cleanup.md`: historical run guide;
- `requirements.txt` and `requirements-cuda.txt`: archived environments; and
- `conftest.py`: historical pytest path/bootstrap support.

Use this package for cross-phase context. Use the owning phase package for exact
configs, source, tests, and result artifacts.
