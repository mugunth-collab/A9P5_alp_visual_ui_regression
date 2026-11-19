# A9P5 code – anchored label propagation (A-LP) for visual UI regression detection

## How to run
    pip install -r requirements.txt
    cd code && python3 run_all.py          # ~9 min on a shared 2-core CPU, all libraries pinned to 1 thread
    dot -Tpng -Gdpi=300 fig1_flowchart.dot -o ../figures/fig1_flowchart.png   # Fig. 1 (Graphviz)

Data are read from `../data/A9P5_visual_ui_regression.csv`. Seeds are fixed (0-4), so results are reproducible.

## What run_all.py does
1. Pre-processing: median imputation + missingness flags (ssim, alignment_deviation_px), winsorising at 0.5/99.5 %,
   change x configuration interaction features, one-hot encoding, standardisation. IDs, build_ts, is_ui_defect and
   defect_type are never used as inputs.
2. Proposed A-LP: class-weighted logistic anchor -> kNN graph on the evidence subspace -> anchored label spreading
   -> LightGBM distillation -> rank fusion (Eqs. (1)-(8) of the paper).
3. Baselines: pixel-diff threshold, SSIM threshold, kNN, supervised LightGBM and logistic regression on the labelled
   subset, plus a fully supervised LightGBM oracle.
4. Label budgets 50/100/200/400/800 x 5 seeds, Wilcoxon tests, ablation, lambda x alpha sensitivity, cross-device test
   (labels from phones only, test on foldables/tablets), per-device and per-defect-type recall.

## Outputs (../results, figures also as 300-dpi PNG in ../figures)
- all_runs.csv – every (method, budget, seed) metric row
- table_main_comparison.csv – Table III (200 labels, mean/std)
- table_label_budget.csv – data of Fig. 2
- table_significance.csv – paired Wilcoxon tests of A-LP vs each baseline
- table_ablation.csv – Table IV
- table_sensitivity_lambda_alpha.csv – data of Fig. 4
- table_cross_device.csv – phones-only labels, test on phones vs foldables+tablets
- table_recall_by_device.csv, table_recall_by_defect_type.csv – data of Fig. 5
- table_dataset_summary.csv – Table I statistics
- metrics.json – all of the above plus config, convergence trace and runtime
- fig2_label_budget, fig3_pr_curves, fig4_sensitivity, fig5_recall_breakdown, fig6_convergence (.jpg)
