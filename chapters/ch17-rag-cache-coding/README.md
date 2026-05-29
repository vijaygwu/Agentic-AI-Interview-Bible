# Chapter 17 Exercise: RAG Cache

Cache public evidence by policy version while rejecting sensitive customer data.

Interview signal: the candidate can distinguish safe retrieval caching from
unsafe response or private-context caching.

Run:

```bash
python3 -B -m pytest -p no:cacheprovider chapters/ch17-rag-cache-coding/tests
```
