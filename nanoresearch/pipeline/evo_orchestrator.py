"""Evo pipeline orchestrator built on the deep 9-stage backbone."""

from __future__ import annotations

from typing import Any

from nanoresearch.pipeline.deep_orchestrator import DeepPipelineOrchestrator
from nanoresearch.schemas.manifest import PipelineMode


class EvoPipelineOrchestrator(DeepPipelineOrchestrator):
    """Deep pipeline with explicit skill, memory, and policy evolution enabled.

    The evo mode intentionally reuses the stable deep stage order and only
    changes the lifecycle contract: adaptive memory and skill evolution are
    required parts of the run, and the manifest records the run as ``evo``.
    """

    _PIPELINE_MODE = PipelineMode.EVO

    def __init__(self, workspace, config, progress_callback=None) -> None:
        # Keep deep behavior intact while making evo runs explicit and auditable.
        config.memory_enabled = True
        config.memory_evolution_enabled = True
        config.skill_evolution_enabled = True
        super().__init__(workspace, config, progress_callback)

    def _get_initial_results(self, topic: str) -> dict[str, Any]:
        results = super()._get_initial_results(topic)
        results["pipeline_mode"] = PipelineMode.EVO.value
        results["evolution_mode"] = {
            "backbone": PipelineMode.DEEP.value,
            "memory_enabled": bool(getattr(self.config, "memory_enabled", True)),
            "memory_evolution_enabled": bool(getattr(self.config, "memory_evolution_enabled", True)),
            "skill_evolution_enabled": bool(getattr(self.config, "skill_evolution_enabled", True)),
            "ram_enabled": bool(getattr(self.config, "ram_enabled", False)),
        }
        return results
