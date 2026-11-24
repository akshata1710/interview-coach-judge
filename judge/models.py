"""Data shapes: the session going in, the rubric, and the evaluation coming out."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------- session in

class Mode(str, Enum):
    solo = "solo"
    ai_assisted = "ai_assisted"
    interviewer = "interviewer"


class Event(BaseModel):
    """One row of session_events. Code snapshots are left out before judging."""
    id: str
    type: Literal["prompt", "ai_reply", "run", "submit"]
    ts: float = Field(description="Seconds since session start")
    text: str = ""


class TestResult(BaseModel):
    name: str
    category: Literal["typical", "edge", "large"]
    hidden: bool
    passed: bool


class Problem(BaseModel):
    title: str
    statement: str
    constraints: str
    optimal_time: str
    optimal_space: str


class Session(BaseModel):
    id: str
    mode: Mode
    language: str
    problem: Problem
    time_limit_s: int
    solve_time_s: int | None = Field(None, description="Time to first all-passing run; None if never")
    final_code: str
    ai_pasted_chars: int = Field(0, description="Chars of final_code that arrived via paste from an AI reply")
    events: list[Event] = []
    tests: list[TestResult] = []

    @property
    def prompts(self) -> list[Event]:
        return [e for e in self.events if e.type == "prompt"]


# ---------------------------------------------------------------- rubric

class Dimension(BaseModel):
    id: str
    name: str
    weight: float
    scored_by: Literal["code", "judge"]
    definition: str | None = None
    anchors: dict[int, str] | None = None

    @model_validator(mode="after")
    def _judge_needs_anchors(self) -> "Dimension":
        if self.scored_by == "judge" and not (self.definition and self.anchors):
            raise ValueError(f"judge dimension {self.id!r} needs a definition and anchors")
        return self


class LabelBand(BaseModel):
    min: float
    label: str


class Rubric(BaseModel):
    version: str
    combined_weights: dict[Mode, dict[Literal["readiness", "prompting"], float]]
    labels: list[LabelBand]
    tracks: dict[Literal["readiness", "prompting"], list[Dimension]]

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "Rubric":
        for track, dims in self.tracks.items():
            total = round(sum(d.weight for d in dims), 6)
            if total != 1.0:
                raise ValueError(f"{track} weights sum to {total}, not 1.0")
        return self

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Rubric":
        path = Path(path) if path else Path(__file__).with_name("rubric.yaml")
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def label_for(self, score: float) -> str:
        for band in self.labels:
            if score >= band.min:
                return band.label
        return self.labels[-1].label


# ---------------------------------------------------------------- judge I/O

class Evidence(BaseModel):
    event_id: str = Field(description="id of a session event, or 'final_code'")
    quote: str = Field(description="Exact words copied from that event or the code")


class Judgment(BaseModel):
    """What the judge model must return for one dimension."""
    score: int = Field(ge=0, le=4)
    evidence: list[Evidence] = Field(min_length=1)
    rationale: str


class PromptRewrite(BaseModel):
    event_id: str
    original: str
    rewritten: str
    why: str


# ---------------------------------------------------------------- evaluation out

class DimensionScore(BaseModel):
    track: Literal["readiness", "prompting"]
    dimension: str
    weight: float
    score: float
    scored_by: Literal["code", "judge"]
    evidence: list[Evidence] = []
    rationale: str
    judge_runs: list[int] = Field([], description="Raw scores from each judge call, for auditing")


class Evaluation(BaseModel):
    session_id: str
    rubric_version: str
    judge_model: str | None
    mode: Mode
    dimensions: list[DimensionScore]
    readiness: float
    prompting: float | None
    combined: float
    label: str
    rewrite: PromptRewrite | None = None
