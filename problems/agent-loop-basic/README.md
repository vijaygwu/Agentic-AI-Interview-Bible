# Bounded Agent Loop

Implement a small, self-contained agent executor for a coding round. The
executor should call a scripted model, route tool calls through a registry, keep
tool-result history, and stop after a fixed number of model steps.

Files:

- `starter.py`: candidate starter with the public contract and TODO methods.
- `solution.py`: reference implementation.
- `tests/test_solution.py`: focused tests for the reference behavior.

Run the focused tests from this directory:

```bash
python3 -B -m pytest -q tests
```

No live LLM or package install is required beyond `pytest`. The problem is
self-contained and uses only the Python standard library at runtime.

