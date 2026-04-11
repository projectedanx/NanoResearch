from __future__ import annotations

from nanoresearch.experiments.deep_persona_runner import (
    DEFAULT_PERSONA_BRIEFS,
    build_assignment_topic,
    build_result_record,
    resolve_variant_runtime_settings,
)


def _question(question_id: str = 'q1') -> dict:
    return {
        'question_id': question_id,
        'domain': 'NLP',
        'difficulty': 'incremental_innovation',
        'background': 'Investigate a lightweight method for improving biomedical QA under strict compute limits.',
        'baselines': ['BioBERT', 'PubMedBERT'],
        'datasets': ['PubMedQA'],
        'user_requirements': 'Generate a novel idea, an executable plan, and keep the design reproducible.',
    }


def test_variant_settings_map_to_expected_runtime_flags() -> None:
    base = resolve_variant_runtime_settings('base_router')
    assert base == {
        'memory_enabled': False,
        'memory_evolution_enabled': False,
        'skill_evolution_enabled': False,
        'same_router_hindsight_sdpo': False,
        'appendix_only': False,
    }

    full = resolve_variant_runtime_settings('full_system')
    assert full == {
        'memory_enabled': True,
        'memory_evolution_enabled': True,
        'skill_evolution_enabled': True,
        'same_router_hindsight_sdpo': True,
        'appendix_only': False,
    }

    appendix = resolve_variant_runtime_settings('context_informed_generation')
    assert appendix['appendix_only'] is True
    assert appendix['same_router_hindsight_sdpo'] is False


def test_build_assignment_topic_includes_persona_and_question_context() -> None:
    assignment = {
        'assignment_id': 'resource_constrained_repro_first::base_router::q1',
        'persona_id': 'resource_constrained_repro_first',
        'variant_name': 'base_router',
        'question': _question(),
    }

    topic = build_assignment_topic(assignment)

    assert 'Persona Profile:' in topic
    assert DEFAULT_PERSONA_BRIEFS['resource_constrained_repro_first'] in topic
    assert 'Research Domain: NLP' in topic
    assert 'Known Baselines: BioBERT; PubMedBERT' in topic
    assert 'Evaluation Datasets: PubMedQA' in topic
    assert 'User Requirements:' in topic


def test_build_result_record_extracts_performance_and_stage_tokens() -> None:
    assignment = {
        'assignment_id': 'persona-a::full_system::q1',
        'persona_id': 'persona-a',
        'variant_name': 'full_system',
        'question': _question(),
    }
    blueprint = {
        'metrics': [
            {'name': 'Accuracy', 'primary': True, 'higher_is_better': True},
            {'name': 'F1', 'primary': False, 'higher_is_better': True},
        ],
        'baselines': [
            {'name': 'BioBERT', 'expected_performance': {'Accuracy': 0.62}},
            {'name': 'PubMedBERT', 'expected_performance': {'Accuracy': 0.65}},
        ],
    }
    experiment_output = {
        'experiment_results': {'Accuracy': 0.74, 'F1': 0.71},
        'experiment_status': 'success',
        'code_execution': {'status': 'success'},
    }
    analysis_output = {
        'analysis': {'final_metrics': {'Accuracy': 0.74, 'F1': 0.71}},
    }
    cost_summary = {
        'stages': {
            'IDEATION': {'total_tokens': 110},
            'PLANNING': {'total_tokens': 220},
            'SETUP': {'total_tokens': 330},
            'CODING': {'total_tokens': 440},
            'EXECUTION': {'total_tokens': 550},
            'ANALYSIS': {'total_tokens': 120},
        },
        'total_tokens': 1770,
    }
    alignment = {'pass_at_1': True, 'feedback': 'Aligned with the request.'}
    novelty = {'novelty_score': 7.5, 'closest_baseline': 'PubMedBERT'}

    record = build_result_record(
        assignment=assignment,
        workspace_path='/tmp/ws',
        blueprint=blueprint,
        experiment_output=experiment_output,
        analysis_output=analysis_output,
        cost_summary=cost_summary,
        alignment_judgment=alignment,
        novelty_judgment=novelty,
        alignment_token_to_pass=330,
    )

    assert record['novelty_score'] == 7.5
    assert record['alignment_pass_at_1'] is True
    assert record['alignment_token_to_pass'] == 330
    assert record['plan_executability'] is True
    assert record['implementation_success'] is True
    assert record['implementation_token_to_runnable'] == 1320
    assert record['total_tokens_from_method_to_code'] == 1650
    assert record['final_performance'] == 0.74
    assert record['baseline_performance'] == 0.65
    assert record['delta_over_baseline'] == 0.09
    assert record['primary_metric_name'] == 'Accuracy'
