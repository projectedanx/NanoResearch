from __future__ import annotations

from pathlib import Path

from nanoresearch.config import ResearchConfig
from nanoresearch.pipeline.evo_orchestrator import EvoPipelineOrchestrator
from nanoresearch.pipeline.state import PipelineStateMachine
from nanoresearch.pipeline.workspace import Workspace
from nanoresearch.schemas.manifest import PipelineMode, PipelineStage, processing_stages_for_mode


def test_evo_uses_deep_stage_order_and_transitions() -> None:
    assert processing_stages_for_mode(PipelineMode.EVO) == processing_stages_for_mode(PipelineMode.DEEP)

    sm = PipelineStateMachine(mode=PipelineMode.EVO)
    for stage in processing_stages_for_mode(PipelineMode.EVO):
        assert sm.can_transition(stage)
        sm.transition(stage)
    assert sm.can_transition(PipelineStage.DONE)


def test_evo_orchestrator_forces_adaptive_evolution_flags(tmp_path: Path) -> None:
    config = ResearchConfig(
        base_url="https://example.com",
        api_key="",
        memory_enabled=False,
        memory_evolution_enabled=False,
        skill_evolution_enabled=False,
    )
    workspace = Workspace.create(
        topic="dry evo pipeline test",
        root=tmp_path,
        pipeline_mode=PipelineMode.EVO,
        config_snapshot=config.snapshot(),
    )

    orchestrator = EvoPipelineOrchestrator(workspace, config)

    assert orchestrator._PIPELINE_MODE == PipelineMode.EVO
    assert config.memory_enabled is True
    assert config.memory_evolution_enabled is True
    assert config.skill_evolution_enabled is True
    assert workspace.manifest.pipeline_mode == PipelineMode.EVO
