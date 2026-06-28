---
name: model-adder
description: Scaffold a new sentiment model following the SOLID registry pattern. Use when adding a model family to the comparison.
tools: Read, Write, Edit, Bash
---

Add a new `SentimentModel` without modifying existing model files.

1. Read `api/models/base.py` for the protocol and `api/models/vader.py` as the reference implementation.
2. Create `api/models/<id>.py` implementing `analyze()` and `explain()`, returning `AnalysisResult`/`Explanation`.
3. Register it in `api/models/registry.py` REGISTRY only — touch no other model file (Open/Closed).
4. Create `tests/test_<id>.py` mirroring `tests/test_vader.py`; mock any remote calls (use `respx` for HTTP, `unittest.mock` for SDK clients).
5. Run `pytest tests/test_<id>.py -v` and confirm green.
