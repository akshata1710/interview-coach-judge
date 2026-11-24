# Interview Coach: scoring judge

Scores one practice session against the rubric in the design doc: coding readiness and prompting quality, 0–4 each, plus one rewritten prompt.

## How a scoring pass runs

1. **Code scorers** (`judge/scorers.py`): correctness, edge cases, and independence/pace come from test results, timing and paste counts. The LLM never scores these.
2. **Judge calls** (`judge/engine.py`): one call per remaining dimension, all in parallel. Each call gets the dimension's definition, its 0/2/4 anchors, the platform facts, and the session fenced as data.
3. **Structured output**: the model must answer through a tool whose input schema is the `Judgment` model (`score`, `evidence[]`, `rationale`).
4. **Evidence check**: every quote must appear verbatim in the event it cites. Prompting dimensions may only cite the candidate's own prompts. A failure triggers one retry that tells the model what was wrong; a second failure raises `JudgeError`.
5. **Double scoring**: each dimension is judged twice. If the runs differ by more than 1, a third run breaks the tie and the median wins. Raw runs are kept in `judge_runs` for auditing.
6. **Rewrite**: one call picks the weakest prompt and rewrites it. The `original` field is always copied from the session, never from the model. A failed rewrite returns `None` without failing the evaluation.

## Files

| Path | What it is |
| --- | --- |
| `judge/rubric.yaml` | Dimensions, weights, anchors, label bands, combined weights per mode. Bump `version` on any change. |
| `judge/models.py` | Pydantic models: `Session` in, `Evaluation` out, and the judge's tool schemas. |
| `judge/prompts.py` | System prompts, request templates, transcript compaction, nonce fencing. |
| `judge/scorers.py` | Deterministic scorers and the facts passed to the judge. |
| `judge/engine.py` | The scoring pass. |
| `judge/llm.py` | `AnthropicClient` (tool-forced, temperature 0, cached system prompt) and `ScriptedClient` for tests. |
| `examples/sample_session.json` | The session from the prototype. |
| `tests/test_engine.py` | 13 tests, no network needed. |

## Run

```bash
pip install -r requirements.txt
pytest -q                                   # uses ScriptedClient, no API key
export ANTHROPIC_API_KEY=...
python run.py examples/sample_session.json  # real scoring with Claude
```

## Using another provider

The engine depends only on the `LLMClient` protocol in `judge/llm.py`: one async `call_tool(system, user, tool) -> dict` method. Any provider that supports forced tool calls or JSON-schema output can implement it.

## Not built yet

- The golden set of hand-scored sessions and the CI check against it (design doc: within ±0.5 on 90% of dimensions).
- The weekly drift monitor.
- Writing results to Postgres (`evaluations`, `dimension_scores`, `prompt_rewrites`).
