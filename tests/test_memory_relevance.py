from nanoresearch.evolution.memory import MemoryScope, MemoryStore, MemoryType


def test_cross_topic_project_memory_is_filtered_from_ideation(tmp_path):
    store = MemoryStore(root=tmp_path, enabled=True, top_k=5)
    store.remember(
        MemoryType.PROJECT_CONTEXT,
        (
            "Evaluation Question ID: pilot_nlp_biomed_qa Research Domain: NLP "
            "PubMedQA BioBERT PubMedBERT healthcare clinical question answering "
            "limited compute reproducible method baseline dataset evaluation ablation results "
        )
        * 80,
        scope=MemoryScope.WORKSPACE_DERIVED,
        source="old-healthcare-run",
        importance=1.0,
        recency_weight=1.5,
        tags=["nlp", "biomed", "qa"],
        project_key="pilot-nlp-biomed-qa",
    )
    store.remember(
        MemoryType.USER_PROFILE,
        "Prefer reproducible single-GPU experiments with clean ablations and no synthetic shortcuts.",
        scope=MemoryScope.GLOBAL_USER,
        source="preference",
        importance=0.8,
        tags=["reproducibility"],
    )
    store.remember(
        MemoryType.PROJECT_CONTEXT,
        (
            "ESM2 adapter fine-tuning for protein fluorescence prediction with LoRA, "
            "frozen encoder ablations, Spearman evaluation, and protein sequence splits."
        ),
        scope=MemoryScope.WORKSPACE_DERIVED,
        source="protein-run",
        importance=0.8,
        tags=["protein", "esm2", "fluorescence"],
        project_key="esm2-fluorescence",
    )

    context = store.render_prompt_context(
        "ideation",
        topic="AI for Science ESM2 LoRA adapters for protein fluorescence prediction",
        tags=["protein", "esm2", "fluorescence"],
        project_key="esm2-fluorescence",
    )

    assert "ESM2" in context or "esm2" in context
    assert "PubMedQA" not in context
    assert "BioBERT" not in context
    assert "healthcare" not in context
    assert "pilot_nlp_biomed_qa" not in context
    assert len(context) < 2200


def test_prompt_memory_context_has_total_budget(tmp_path):
    store = MemoryStore(root=tmp_path, enabled=True, top_k=10)
    for index in range(8):
        store.remember(
            MemoryType.PROJECT_CONTEXT,
            (
                f"ESM2 protein fluorescence LoRA adapter experiment note {index}. "
                "This deliberately long record should be compacted before entering prompts. "
            )
            * 60,
            scope=MemoryScope.WORKSPACE_DERIVED,
            source=f"protein-run-{index}",
            importance=0.7,
            tags=["protein", "esm2", "fluorescence"],
            project_key="esm2-fluorescence",
        )

    context = store.render_prompt_context(
        "ideation",
        topic="ESM2 protein fluorescence LoRA adapter prediction",
        tags=["protein", "esm2", "fluorescence"],
        project_key="esm2-fluorescence",
        top_k=10,
    )

    assert len(context) < 2300
    assert "[trimmed]" in context

def test_project_scoped_decision_history_is_filtered_across_topics(tmp_path):
    store = MemoryStore(root=tmp_path, enabled=True, top_k=5)
    store.remember(
        MemoryType.DECISION_HISTORY,
        (
            "Key gaps for lightweight tabular classification with random forest, "
            "accuracy, feature count, ablation, and benchmark complexity metrics. "
        )
        * 40,
        scope=MemoryScope.WORKSPACE_DERIVED,
        source="old-tabular-run",
        importance=1.0,
        recency_weight=1.5,
        tags=["tabular", "classification"],
        project_key="tabular-feature-selection",
    )

    context = store.render_prompt_context(
        "literature",
        topic="ESM2 protein fluorescence LoRA adapter prediction",
        tags=["protein", "esm2", "fluorescence"],
        project_key="esm2-fluorescence",
    )

    assert context == ""

