"""Deterministic scorers: anything measurable in code never goes to the LLM."""
from __future__ import annotations

from .models import Mode, Session


def _band(value: float, bands: list[tuple[float, int]]) -> int:
    """First score whose threshold `value` meets, checked top-down (bands sorted high to low)."""
    for threshold, score in bands:
        if value >= threshold:
            return score
    return 0


def correctness(session: Session) -> tuple[float, str]:
    hidden = [t for t in session.tests if t.hidden]
    if not hidden:
        return 0.0, "No hidden tests ran."
    passed = sum(t.passed for t in hidden)
    rate = passed / len(hidden)
    score = 4 if rate == 1 else _band(rate, [(0.9, 3), (0.7, 2), (0.3, 1)])
    return float(score), f"{passed} of {len(hidden)} hidden tests passed."


def edge_cases(session: Session) -> tuple[float, str]:
    edge = [t for t in session.tests if t.category == "edge"]
    if not edge:
        return 4.0, "Problem has no edge-tagged tests."
    failed = [t.name for t in edge if not t.passed]
    rate = 1 - len(failed) / len(edge)
    score = 4 if not failed else _band(rate, [(0.75, 3), (0.5, 2), (0.01, 1)])
    note = "Passed every edge test." if not failed else f"Failed: {', '.join(failed)}."
    return float(score), note


def independence_pace(session: Session) -> tuple[float, str]:
    if session.solve_time_s is None:
        time_score, time_note = 0, "never reached an all-passing run"
    else:
        ratio = session.solve_time_s / session.time_limit_s
        time_score = 4 if ratio <= 0.6 else 3 if ratio <= 0.8 else 2 if ratio <= 1.0 else 0
        time_note = f"solved in {session.solve_time_s // 60} of {session.time_limit_s // 60} minutes"

    if session.mode == Mode.solo:
        return float(time_score), time_note[0].upper() + time_note[1:] + "."

    share = session.ai_pasted_chars / max(len(session.final_code), 1)
    share_score = 4 if share <= 0.2 else 3 if share <= 0.4 else 2 if share <= 0.6 else 1 if share <= 0.8 else 0
    score = round((time_score + share_score) / 2 * 2) / 2  # nearest 0.5
    return score, f"{time_note[0].upper() + time_note[1:]}; {share:.0%} of final code pasted from the assistant."


SCORERS = {
    "correctness": correctness,
    "edge_cases": edge_cases,
    "independence_pace": independence_pace,
}


def facts_for_judge(session: Session) -> dict[str, str]:
    """Platform-measured facts the judge sees as context (it never re-derives these)."""
    return {
        "Optimal complexity": f"time {session.problem.optimal_time}; space {session.problem.optimal_space}",
        "Hidden tests": correctness(session)[1],
        "Edge tests": edge_cases(session)[1],
        "Mode": session.mode.value,
    }
