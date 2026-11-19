# A9P5 – An Automated Visual Regression Detection Framework for Identifying UI Defects Across Heterogeneous Mobile Devices and Display Configurations

Proposed method: anchored semi-supervised label propagation (A-LP) for visual defect detection with few labels.
Baselines: pixel-diff threshold, SSIM threshold, kNN, supervised-only LightGBM / logistic regression on the labelled subset.

Contents
- paper/   – IEEE conference paper (.docx and .pdf, 6 pages)
- data/    – dataset CSV (16,000 synthetic screenshot comparisons) and its README / data dictionary
- code/    – run_all.py (full pipeline), fig1_flowchart.dot (Fig. 1), requirements.txt, README.md
- results/ – all figures (JPG), all tables (CSV), metrics.json

Reproduce (about 5–9 min on a 2-core CPU):
    pip install -r code/requirements.txt
    cd code && python3 run_all.py
