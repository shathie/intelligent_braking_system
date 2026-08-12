# Claim to Evidence Map (Resubmission)

Student: [Your Name]  
Project: Intelligent Braking System  
Date: [DD MMM YYYY]  
Git Repo: [Repo URL]  
Primary Run Tag: [Run ID / Timestamp]

## Purpose

This page maps each claim in the report/slides to direct, reproducible artifacts from the current run cycle (logs, metrics, plots, reports, and commits).

## Claim Mapping Table

| # | Claim (exact wording used in report/slides) | Evidence Type | Exact Evidence Location | Reproduction Pointer | Verification Status |
|---|---|---|---|---|---|
| 1 | ViT road-surface classification weighted F1 = [X.XX]. | Metrics JSON + confusion matrix + run log | output/metrics/vit_metrics.json; output/plots/confusion_matrix.png; output/logs/env_cpuopt_e50_memsafe_20260730_180724.txt | Command list in `Appendix A` + commit [SHA] | Verified |
| 2 | Temporal model improves sequence understanding over frame-only baseline. | Comparative metrics + training curves | output/metrics/temporal_metrics.json; output/plots/temporal_network_training_curve_20260809_234002.png | Baseline vs temporal commands in `Appendix A` | Verified |
| 3 | Fusion network outperforms single-modality models on overall classification metrics. | Per-model metrics + summary table | output/metrics/fusion_metrics.json; output/reports/full_report.html | Evaluation script/config in `Appendix B` | Verified |
| 4 | PINN friction estimation achieves reported MAE/RMSE values. | Regression metrics + friction plot | output/metrics/pinn_metrics.json; output/plots/friction_prediction.png | PINN steps in `Appendix A` | Verified |
| 5 | SAC controller produces stable braking policy in simulation. | SAC metrics + reward curve + simulation figure | output/metrics/sac_metrics.json; output/plots/sac_training_reward_20260809_234003.png; output/plots/braking_simulation.png | Simulation command in `Appendix A` | Verified |
| 6 | Dataset composition and class-imbalance handling are documented and applied. | Dataset report + imbalance plot + scripts | output/reports/dataset_analysis_report.html; output/reports/class_imbalance_improvements.png; scripts/analyze_class_imbalance.py | Data prep + analysis steps in `Appendix A` | Verified |
| 7 | Final report/slides use only current-run outputs from this resubmission cycle. | Timestamped artifacts + commit trace | output/training_history_20260808_153957.json; output/evaluation_report.html | Commit list in `Appendix C` | Verified |

## Appendix A: Exact Commands Used

- Add every command actually executed, in order, with date/time.
- Include environment details: OS, Python version, and dependency snapshot.

## Appendix B: Config Provenance

List config files used per run:

- configs/vit_config.yaml
- configs/temporal_config.yaml
- configs/fusion_config.yaml
- configs/pinn_config.yaml
- configs/control_config.yaml

## Appendix C: Git Traceability

For each relevant commit, include:

1. Commit SHA
2. Commit date/time
3. Commit message
4. Claims supported

## Declaration Sentence (Optional for Report)

All quantitative and qualitative claims in this submission are mapped to direct, reproducible artifacts generated from the current run cycle. No claim is supported only through emails or external explanations.
