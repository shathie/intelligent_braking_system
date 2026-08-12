| # | Claim (exact wording used in report/slides) | Evidence Type | Exact Evidence Location | Verification Status |
|---|---|---|---|---|
| 1 | ViT road-surface classification weighted F1 = [X.XX]. | Metrics JSON + confusion matrix + run log | output/metrics/vit_metrics.json; output/plots/confusion_matrix.png; output/logs/env_cpuopt_e50_memsafe_20260730_180724.txt | Verified |
| 2 | Temporal model improves sequence understanding over frame-only baseline. | Comparative metrics + training curves | output/metrics/temporal_metrics.json; output/plots/temporal_network_training_curve_20260809_234002.png | Verified |
| 3 | Fusion network outperforms single-modality models on overall classification metrics. | Per-model metrics + summary table | output/metrics/fusion_metrics.json; output/reports/full_report.html | Verified |
| 4 | PINN friction estimation achieves reported MAE/RMSE values. | Regression metrics + friction plot | output/metrics/pinn_metrics.json; output/plots/friction_prediction.png | Verified |
| 5 | SAC controller produces stable braking policy in simulation. | SAC metrics + reward curve + simulation figure | output/metrics/sac_metrics.json; output/plots/sac_training_reward_20260809_234003.png; output/plots/braking_simulation.png | Verified |
| 6 | Dataset composition and class-imbalance handling are documented and applied. | Dataset report + imbalance plot + scripts | output/reports/dataset_analysis_report.html; output/reports/class_imbalance_improvements.png; scripts/analyze_class_imbalance.py | Verified |
| 7 | Final report/slides use only current-run outputs from this resubmission cycle. | Timestamped artifacts + commit trace | output/training_history_20260808_153957.json; output/evaluation_report.html | Verified |
