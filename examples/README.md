# Original benchmark experiment entry points

Run from the repository root with the benchmark environment active:

```bash
PYTHONPATH=. python examples/easy-example-experiment.py
```

- `easy-example-experiment.py`: compact upstream usage example.
- `example-experiment.py`: more detailed upstream example.
- `mission1_experiments.py`, `mission2_experiments.py`: original mission benchmark
  runners restored from the repository's HEAD for reference.

These launch benchmark work and may require Docker and datasets. For the thesis
Transformer experiments use `python -m esa_thesis train` instead.
