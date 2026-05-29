# Chapter 14 Exercise: Structured Outputs

Validate a model-produced refund decision before the agent can act on it.

Interview signal: the candidate treats model output as untrusted data, not as
already-typed application state.

Run:

```bash
python3 -B -m pytest -p no:cacheprovider chapters/ch14-structured-outputs/tests
```
