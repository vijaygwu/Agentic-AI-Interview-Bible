# Chapter 24 Exercise: Agent Traces

Record safe trace spans for a refund task without storing raw prompts, secrets,
or hidden reasoning.

Interview signal: the candidate can debug agent behavior while respecting data
minimization.

Run:

```bash
python3 -B -m pytest -p no:cacheprovider chapters/ch24-observability/tests
```
