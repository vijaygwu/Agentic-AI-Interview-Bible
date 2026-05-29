# Chapter 15 Exercise: Reliability Guards

Wrap a flaky dependency with a retry budget and circuit breaker.

Interview signal: the candidate can bound retries and stop repeatedly calling a
failing dependency.

Run:

```bash
python3 -B -m pytest -p no:cacheprovider chapters/ch15-reliability-coding/tests
```
