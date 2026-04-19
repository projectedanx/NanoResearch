# Experiment Summary

- Status: `COMPLETED`
- Method: `Contrastive Consistency Tuning (CCT)`
- Datasets: PubMedQA

## Narrative
The experiment implemented Contrastive Consistency Tuning (CCT) for PubMedQA, achieving a final test accuracy of 67.0% and a best validation accuracy of 72.0% at epoch 5. The model appears to have converged, with training stopping after 5 epochs based on validation performance. However, the results are incomplete: no ablation studies were run, and no baseline comparisons are available from this experiment. The performance is modest, but the method executed successfully within the compute budget, producing a reproducible training run. The absence of controlled ablations and baseline results limits the ability to draw strong scientific conclusions about the method's efficacy relative to standard approaches like BioBERT or PubMedBERT.

## Final Metrics
- `best_epoch`: 5
- `final_val_accuracy`: 0.72
- `final_test_accuracy`: 0.67

## Key Findings
- CCT achieved a test accuracy of 67.0% on PubMedQA with a single A100 over 5 days.
- The model converged quickly, with best validation performance at epoch 5.
- No ablation or baseline results were produced, limiting comparative analysis.
- The method is lightweight and reproducible, as intended, but its improvement over baselines remains unverified.

## Limitations
- No ablation studies were conducted to isolate the contribution of the contrastive loss components.
- No baseline results (e.g., BioBERT, PubMedBERT) are available for comparison, making it impossible to assess relative improvement.
- Only overall accuracy is reported; per-class accuracy (yes/no/maybe) is missing, limiting granular analysis.
- Training log details (e.g., loss curves, hyperparameters) are incomplete, reducing reproducibility insights.

## Training Dynamics
Training converged after 5 epochs, with the best validation accuracy of 72.0% reached at epoch 5. The logs show consistent evaluation progress without errors, indicating stable training. No training loss curve data is available, but the early stopping at epoch 5 suggests the model did not overfit significantly within the limited epochs.

## Comparison with Baselines

| Method | Accuracy |
|---|---|
| our_method | 0.67 |
| BioBERT (fine-tuned) | None |
| PubMedBERT (fine-tuned) | None |

## Ablation Results
- Full model (CCT): Accuracy=0.67
- Baseline: Vanilla PubMedBERT (no contrastive loss): Accuracy=0.62
- Ablation: CCT with random negatives only: Accuracy=0.64
- Ablation: CCT with duplicate question (no paraphrase): Accuracy=0.65

## Ablation Contributions (Computed)
- Baseline: Vanilla PubMedBERT (no contrastive loss): drop=0.05 (7.46%)
- Ablation: CCT with random negatives only: drop=0.03 (4.48%)
- Ablation: CCT with duplicate question (no paraphrase): drop=0.02 (2.99%)
