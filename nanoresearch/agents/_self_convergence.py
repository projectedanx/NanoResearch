"""Generic self-convergence loop helper.

Factored out from ``IdeationAgent._run_search_loop`` so the same
"LLM self-eval coverage_score >= threshold OR max_rounds, refine on gaps"
pattern can be reused by other agents (writing, ...).

Reference implementation in ideation:

    for _eval_round in range(2):                              # max 2 rounds
        coverage = await self._evaluate_search_coverage(...)
        score = coverage.get("coverage_score", 10)
        if score >= 8:                                        # threshold
            break
        missing = coverage.get("missing_directions", [])
        if not missing:
            break
        new_papers = await self._supplementary_search(missing, ...)
        if new_papers:
            papers.extend(new_papers)

This module turns that pattern into a reusable async helper so the same
shape can apply to section-quality refinement in writing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")  # the artifact under refinement (e.g. section text, paper list)


@dataclass
class ConvergenceEval:
    """Outcome of one round's self-evaluation."""

    score: int                       # 1..10
    missing: list[str] = field(default_factory=list)  # gap descriptions
    reason: str = ""


@dataclass
class ConvergenceTrace:
    """Per-round trace, mainly for logging / debugging."""

    rounds: int = 0
    final_score: int = 0
    converged: bool = False  # True if score >= threshold; False if max_rounds hit
    history: list[ConvergenceEval] = field(default_factory=list)


# Default convergence parameters mirror the ideation reference.
DEFAULT_THRESHOLD = 8       # score out of 10
DEFAULT_MAX_ROUNDS = 2      # ideation runs at most 2 eval rounds


async def converge(
    *,
    artifact: T,
    eval_fn: Callable[[T], Awaitable[ConvergenceEval]],
    refine_fn: Callable[[T, ConvergenceEval], Awaitable[T]],
    threshold: int = DEFAULT_THRESHOLD,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    label: str = "convergence",
) -> tuple[T, ConvergenceTrace]:
    """Iteratively self-evaluate and refine ``artifact`` until quality plateaus.

    Args:
        artifact: The thing being refined (text, list of papers, etc.). Mutable
            or immutable — the refine function returns an updated copy.
        eval_fn: ``async (artifact) -> ConvergenceEval``. Returns a 1-10 score
            plus a list of gap descriptions. Should be fail-open: on internal
            failure return a high score so the loop terminates rather than spins.
        refine_fn: ``async (artifact, eval_result) -> artifact``. Uses the gap
            list to produce an improved artifact. Returning the original
            unchanged signals "no progress possible" and ends the loop.
        threshold: Score at which the loop is considered converged (default 8).
        max_rounds: Hard cap on rounds (default 2). The loop runs at most this
            many evaluations *including* the first one.
        label: Used in log messages.

    Returns:
        ``(refined_artifact, trace)``.

    The loop terminates as soon as ANY of:
        - score >= threshold
        - eval returns no missing items (nothing actionable to refine on)
        - refine_fn returns the artifact unchanged (nothing improved)
        - max_rounds reached
    """
    trace = ConvergenceTrace()
    current = artifact

    for round_idx in range(max_rounds):
        trace.rounds = round_idx + 1
        try:
            eval_result = await eval_fn(current)
        except Exception as exc:
            logger.warning(
                "[%s round %d] eval_fn crashed (%s) — terminating loop fail-open",
                label, round_idx + 1, exc,
            )
            break

        trace.history.append(eval_result)
        trace.final_score = eval_result.score
        logger.info(
            "[%s round %d] score=%d/%d, missing=%d",
            label, round_idx + 1, eval_result.score, threshold,
            len(eval_result.missing),
        )

        if eval_result.score >= threshold:
            trace.converged = True
            break
        if not eval_result.missing:
            # No actionable gaps -- nothing to refine on.
            break
        if round_idx == max_rounds - 1:
            # Last round: skip the refine step (we won't re-evaluate anyway).
            break

        try:
            refined = await refine_fn(current, eval_result)
        except Exception as exc:
            logger.warning(
                "[%s round %d] refine_fn crashed (%s) — keeping previous artifact",
                label, round_idx + 1, exc,
            )
            break

        # No-progress guard: if refine returns the same object, stop.
        if refined is current:
            break
        current = refined

    return current, trace
