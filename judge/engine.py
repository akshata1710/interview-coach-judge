"""Scoring pass: deterministic scorers first, then one judge call per dimension (in parallel),
double-scored with a tie-break, evidence checked against the session, then the prompt rewrite."""
from __future__ import annotations

import asyncio
import re
import statistics
from typing import Literal

from pydantic import ValidationError

from . import prompts, scorers
from .llm import LLMClient
from .models import (
    Dimension, DimensionScore, Evaluation, Judgment, Mode, PromptRewrite, Rubric, Session,
)

Track = Literal["readiness", "prompting"]
MAX_ATTEMPTS = 3        # retries when the output fails schema or evidence checks
DISAGREEMENT = 1        # two runs further apart than this trigger a third


class JudgeError(RuntimeError):
    pass


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def check_evidence(j: Judgment, session: Session, track: Track) -> list[str]:
    """Every quote must exist verbatim in the event it cites. Returns problems found."""
    texts = {e.id: e.text for e in session.events} | {"final_code": session.final_code}
    allowed = {i: texts[i] for i in prompts.allowed_evidence_ids(session, track)}

    problems = []
    for ev in j.evidence:
        source = allowed.get(ev.event_id)
        if source is None:
            problems.append(f"event_id {ev.event_id!r} is not a valid {track} source")
        elif _norm(ev.quote) not in _norm(source):
            problems.append(f"quote {ev.quote!r} does not appear in {ev.event_id}")
    return problems


async def _one_valid_judgment(
    client: LLMClient, dim: Dimension, track: Track, session: Session, facts: dict[str, str]
) -> Judgment:
    user = prompts.render_dimension_request(dim, session, facts, track)
    last_error = ""
    for _ in range(MAX_ATTEMPTS):
        msg = user if not last_error else (
            f"{user}\n\nYour previous answer was rejected: {last_error}. "
            f"Quote exactly, and cite only these event_ids: "
            f"{', '.join(prompts.allowed_evidence_ids(session, track))}."
        )
        raw = await client.structured(prompts.JUDGE_SYSTEM, msg, Judgment)
        try:
            judgment = Judgment.model_validate(raw)
        except ValidationError as exc:
            last_error = f"schema error: {exc.errors()[0]['msg']}"
            continue
        problems = check_evidence(judgment, session, track)
        if not problems:
            return judgment
        last_error = "; ".join(problems)
    raise JudgeError(f"{dim.id}: no valid judgment after {MAX_ATTEMPTS} attempts ({last_error})")


async def judge_dimension(
    client: LLMClient, dim: Dimension, track: Track, session: Session, facts: dict[str, str]
) -> DimensionScore:
    results = await asyncio.gather(
        *(_one_valid_judgment(client, dim, track, session, facts) for _ in range(2)),
        return_exceptions=True,
    )
    runs = [r for r in results if isinstance(r, Judgment)]
    if not runs:  # both runs failed every attempt
        raise next(r for r in results if isinstance(r, BaseException))
    if len(runs) == 1 or abs(runs[0].score - runs[1].score) > DISAGREEMENT:
        try:
            runs.append(await _one_valid_judgment(client, dim, track, session, facts))
        except JudgeError:
            if len(runs) == 1:
                pass  # keep the one valid run rather than fail the session
            else:
                raise

    final = statistics.median(r.score for r in runs)
    # report the evidence from the run closest to the final score
    chosen = min(runs, key=lambda r: abs(r.score - final))
    return DimensionScore(
        track=track, dimension=dim.id, weight=dim.weight, score=float(final), scored_by="judge",
        evidence=chosen.evidence, rationale=chosen.rationale, judge_runs=[r.score for r in runs],
    )


def score_in_code(dim: Dimension, track: Track, session: Session) -> DimensionScore:
    score, note = scorers.SCORERS[dim.id](session)
    return DimensionScore(
        track=track, dimension=dim.id, weight=dim.weight, score=score, scored_by="code", rationale=note,
    )


async def make_rewrite(
    client: LLMClient, session: Session, prompting: list[DimensionScore]
) -> PromptRewrite | None:
    """Best effort: a bad rewrite should never sink the whole evaluation."""
    if not session.prompts:
        return None
    scores = {d.dimension: d.score for d in prompting}
    try:
        raw = await client.structured(
            prompts.REWRITE_SYSTEM, prompts.render_rewrite_request(session, scores), PromptRewrite
        )
        rewrite = PromptRewrite.model_validate(raw)
    except (ValidationError, RuntimeError):
        return None
    original = {p.id: p.text for p in session.prompts}.get(rewrite.event_id)
    if original is None:
        return None
    return rewrite.model_copy(update={"original": original})  # never trust the model's copy


def _weighted(dims: list[DimensionScore]) -> float:
    return round(sum(d.weight * d.score for d in dims), 1)


async def evaluate(session: Session, client: LLMClient, rubric: Rubric | None = None) -> Evaluation:
    rubric = rubric or Rubric.load()
    facts = scorers.facts_for_judge(session)
    tracks: list[Track] = ["readiness"] if session.mode == Mode.solo else ["readiness", "prompting"]

    pending = []
    for track in tracks:
        for dim in rubric.tracks[track]:
            if dim.scored_by == "code":
                pending.append(asyncio.sleep(0, result=score_in_code(dim, track, session)))
            else:
                pending.append(judge_dimension(client, dim, track, session, facts))
    dims: list[DimensionScore] = list(await asyncio.gather(*pending))

    readiness = _weighted([d for d in dims if d.track == "readiness"])
    prompting_dims = [d for d in dims if d.track == "prompting"]
    prompting = _weighted(prompting_dims) if prompting_dims else None

    w = rubric.combined_weights[session.mode]
    combined = round(w["readiness"] * readiness + w["prompting"] * (prompting or 0.0), 1)

    rewrite = await make_rewrite(client, session, prompting_dims) if prompting_dims else None

    return Evaluation(
        session_id=session.id, rubric_version=rubric.version, judge_model=client.model,
        mode=session.mode, dimensions=dims, readiness=readiness, prompting=prompting,
        combined=combined, label=rubric.label_for(combined), rewrite=rewrite,
    )
