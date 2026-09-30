# Benchmarking & Evaluation

Run the deterministic Theme 4 replay:

```bash
python scripts/benchmark_theme4.py
```

The replay reads cases from `data/evaluation/streaming_cases.json`; queries are
not embedded in application code.

The report covers:
- G2 early-retrieval rate among eligible streaming queries
- G3 multi-intent identification rate
- G4 citation/grounding support
- G5 refinement without session reset
- G6 trace coverage

For architecture ablations, run:

```bash
python scripts/run_ablations.py
```

The script compares dense-only retrieval with the existing hybrid path and
compares the production controller with a deliberately simple always-retrieve
baseline. The baseline is a measurement aid only and is not used by the
application.
