# Agentic AI Interview Bible Companion Code

Runnable companion code for *The AI Agent Engineer Interview Bible*.

This repository is organized around the coding, system-design, evaluation,
observability, and incident-response exercises in Book 5 of the Agentic AI
series. The code is intentionally small enough to discuss in interviews, but
structured like production software: typed interfaces, bounded execution,
explicit timeouts, deterministic tests, and clear failure modes.

## Layout

```text
problems/
  agent-loop-basic/
chapters/
  ch13-agent-loop/
  ch14-structured-outputs/
  ch15-reliability-coding/
  ch16-evaluation-coding/
  ch17-rag-cache-coding/
  ch24-observability/
  ch25-cost-latency-scaling/
  ch26-testing-release/
src/
  agentic_interview_bible/
tests/
```

Each chapter folder has a short README, an interview prompt in `exercise.py`, a
complete `solution.py`, and a deterministic pytest file under `tests/`.

The `problems/` directory holds the fully runnable problem folder
`agent-loop-basic/` (README, `starter.py`, `solution.py`, and local tests) for
timed practice: read the README, start from `starter.py`, compare against
`solution.py`, and run the local tests. The remaining problems in the book
present their solutions inline in the text and reuse the primitives in
`agentic_interview_bible`, so you work them against the package rather than a
per-problem scaffold.

## First Exercises

- Build a bounded agent loop.
- Add a typed tool registry.
- Add max-step protection.
- Add structured output validation.
- Add retry budgets and circuit breakers.
- Add a deterministic MockLLM test harness.
- Add a minimal evaluation runner.
- Add safe retrieval caching, trace records, task budgets, and release gates.

The root package exposes the core interview primitives:

- `AgentExecutor`, `AgentStep`, `ToolCall`, `ToolRegistry`, and typed executor
  errors for agent-loop coding rounds.
- `StructuredOutputValidator` and `validate_refund_decision` for schema checks.
- `RetryBudget` and `CircuitBreaker` for reliability exercises.
- `MockLLM` for deterministic tests that can assert tool-result history.
- `run_eval_cases` and `decide_release` for evaluation and release gates.
- `EvidenceCache`, `TaskBudget`, and `InMemoryTraceSink` for RAG, cost, and
  observability drills.

Run everything locally:

```bash
pip install -e ".[dev]"   # pytest ships as the 'dev' extra
python3 -B -m pytest -p no:cacheprovider
```

## External Repository

The canonical companion repository is:

https://github.com/vijaygwu/Agentic-AI-Interview-Bible
