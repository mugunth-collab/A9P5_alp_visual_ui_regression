#!/usr/bin/env python3
"""A9P5 - Anchored semi-supervised label propagation (A-LP) for visual UI regression detection.

Runs everything: preprocessing, proposed method, baselines, label-budget study, ablation,
sensitivity grid, cross-device test, figures. Outputs go to ../results and ../figures.
Usage:  cd code && python3 run_all.py
"""
import os, json, time, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import rankdata, wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score as AP, roc_auc_score, precision_recall_curve,
                             f1_score, precision_score, recall_score)
import lightgbm as lgb
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'A9P5_visual_ui_regression.csv')
RES = os.path.join(HERE, '..', 'results'); FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
N_TEST = 3200
NL_MAIN = 200
BUDGETS = [50, 100, 200, 400, 800]
CFG = dict(k=15, alpha=0.8, lam=5.0, iters=40, C=0.5)

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.titlesize': 8})

# ---------------------------------------------------------------- 1. preprocessing
d = pd.read_csv(DATA)
y = d['is_ui_defect'].values.astype(int)
# not used as features: IDs, timestamp, and both targets (is_ui_defect, defect_type)
NUM = ['os_versions_behind', 'font_scale', 'dark_mode', 'rtl_locale', 'component_count',
       'style_change_in_build', 'layout_change_in_build', 'string_change_in_build', 'dynamic_content_ratio',
       'has_animation', 'pixel_diff_pct', 'ssim', 'diff_region_count', 'largest_diff_region_pct',
       'text_overflow_count', 'bbox_overlap_count', 'alignment_deviation_px', 'contrast_ratio_drop',
       'element_count_delta']
X = d[NUM].copy()
miss = {c: float(X[c].isna().mean()) for c in X}
for c in ['ssim', 'alignment_deviation_px']:
    X[c + '_miss'] = X[c].isna().astype(float)
    X[c] = X[c].fillna(X[c].median())
clip_counts = {}
for c in NUM:  # winsorise injected outliers at the 0.5 / 99.5 percentiles
    lo, hi = X[c].quantile([.005, .995])
    clip_counts[c] = int(((X[c] < lo) | (X[c] > hi)).sum())
    X[c] = X[c].clip(lo, hi)
narrow = ((d.device_class == 'foldable_android') | (d.orientation == 'landscape')).astype(float)
X['abs_elem_delta'] = X.element_count_delta.abs()
X['layout_x_font'] = X.layout_change_in_build * X.font_scale
X['string_x_rtl_font'] = X.string_change_in_build * (X.rtl_locale + (X.font_scale >= 1.5))
X['style_x_dark'] = X.style_change_in_build * X.dark_mode
X['layout_x_narrow'] = X.layout_change_in_build * narrow
X = pd.concat([X, pd.get_dummies(d[['device_class', 'orientation', 'screen_type', 'platform']]).astype(float)], axis=1)
COLS = list(X.columns)
Xs = StandardScaler().fit_transform(X.values)
EVID = ['text_overflow_count', 'bbox_overlap_count', 'alignment_deviation_px', 'contrast_ratio_drop',
        'abs_elem_delta', 'layout_x_font', 'string_x_rtl_font', 'style_x_dark', 'layout_x_narrow',
        'largest_diff_region_pct', 'diff_region_count']
EI = [COLS.index(c) for c in EVID]
N = len(d)

# graph construction is label-independent -> cache per feature space
_graph = {}
def graph(name, Z, k):
    key = (name, k)
    if key not in _graph:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        dist, ind = nn.kneighbors(Z)
        dist, ind = dist[:, 1:], ind[:, 1:]
        sig = np.median(dist[:, -1]) + 1e-9
        w = np.exp(-dist ** 2 / sig ** 2)                         # Eq. (1)
        W = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(N), k), ind.ravel())), shape=(N, N))
        W = W.maximum(W.T)
        dg = np.asarray(W.sum(1)).ravel()
        Dm = sp.diags(1 / np.sqrt(dg))
        _graph[key] = (Dm @ W @ Dm).tocsr()                        # Eq. (2)
    return _graph[key]

def spread(S, lab, yl, prior=None, lam=0.0, alpha=0.8, iters=40, trace=None):
    Y0 = np.zeros((N, 2)); Y0[lab, yl] = 1.0; Y0 /= Y0.sum(0, keepdims=True)   # class-balanced seeds, Eq. (3)
    if prior is not None and lam > 0:
        P = np.c_[1 - prior, prior]; P[lab] = 0; P /= P.sum(0, keepdims=True); Y0 = Y0 + lam * P  # Eq. (4)
    F = Y0.copy()
    for t in range(iters):
        Fn = alpha * (S @ F) + (1 - alpha) * Y0                       # Eq. (5)
        if trace is not None: trace.append(float(np.abs(Fn - F).sum() / np.abs(F).sum()))
        F = Fn
    return F[:, 1] / (F.sum(1) + 1e-12)                               # Eq. (6)

def anchor(lab, yl, C=0.5):
    return LogisticRegression(max_iter=3000, C=C, class_weight='balanced').fit(Xs[lab], yl).predict_proba(Xs)[:, 1]

def distil(pool, lab, soft, seed):
    tgt = soft[pool].copy(); m = np.isin(pool, lab); tgt[m] = y[pool[m]]
    g = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
                          verbose=-1, n_jobs=1).fit(Xs[pool], tgt)                  # Eq. (7)
    return g.predict(Xs)

def rnk(v): return rankdata(v) / len(v)

def alp(lab, yl, pool, seed, k=CFG['k'], alpha=CFG['alpha'], lam=CFG['lam'], space='evid',
        use_prior=True, use_distil=True, use_fuse=True, trace=None):
    """Proposed anchored label propagation. Returns a score for every node."""
    pa = anchor(lab, yl)
    Z = Xs[:, EI] if space == 'evid' else Xs
    S = graph(space, Z, k)
    ps = spread(S, lab, yl, pa if use_prior else None, lam if use_prior else 0.0, alpha, CFG['iters'], trace)
    out = distil(pool, lab, ps, seed) if use_distil else ps
    return 0.5 * rnk(out) + 0.5 * rnk(pa) if use_fuse else out          # Eq. (8)

# ---------------------------------------------------------------- baselines
def b_pixel(lab, yl): return X['pixel_diff_pct'].values.astype(float)
def b_ssim(lab, yl): return -X['ssim'].values.astype(float)
def b_sup_gbm(lab, yl, seed):
    g = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                           class_weight='balanced', random_state=seed, verbose=-1, n_jobs=1).fit(Xs[lab], yl)
    return g.predict_proba(Xs)[:, 1]
def b_sup_lr(lab, yl): return anchor(lab, yl)
def b_knn(lab, yl):
    return KNeighborsClassifier(n_neighbors=7, weights='distance').fit(Xs[lab], yl).predict_proba(Xs)[:, 1]
def b_full(tr, seed):
    g = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=seed,
                           verbose=-1, n_jobs=1).fit(Xs[tr], y[tr])
    return g.predict_proba(Xs)[:, 1]

# ---------------------------------------------------------------- metrics
def metrics(yt, s, lab_scores, lab_y):
    """Threshold for F1 is set without test labels: flag the top fraction equal to the
    defect prevalence estimated from the labelled set."""
    prev = max(lab_y.mean(), 1.0 / len(lab_y))
    thr = np.quantile(s, 1 - prev)
    pred = (s >= thr).astype(int)
    pr, rc, th = precision_recall_curve(yt, s)
    ok = rc >= 0.90
    p90 = float(pr[ok].max())
    # false alarms per 1000 comparisons at 90 % recall
    i = np.where(ok)[0]; j = i[np.argmax(pr[i])]
    tp = rc[j] * yt.sum(); fp = tp / max(pr[j], 1e-9) - tp
    return dict(pr_auc=AP(yt, s), roc_auc=roc_auc_score(yt, s), f1=f1_score(yt, pred),
                prec=precision_score(yt, pred, zero_division=0), rec=recall_score(yt, pred),
                p_at_r90=p90, fa_per_1k=1000 * fp / len(yt))

def split(seed):
    rng = np.random.RandomState(seed); idx = rng.permutation(N)
    return idx[:N_TEST], idx[N_TEST:], rng

def draw_labels(rng, pool, nl):
    # stratification is NOT used (labels are drawn uniformly, as a tester would); ensure >=2 positives
    while True:
        lab = rng.choice(pool, nl, replace=False)
        if y[lab].sum() >= 2: return lab

# ---------------------------------------------------------------- 2. main + label-budget study
METHODS = ['Pixel-diff thr.', 'SSIM thr.', 'kNN (labelled)', 'Sup. GBM (labelled)', 'Sup. LR (labelled)',
           'A-LP (proposed)']
rows = []; pr_store = {}; per_dev = []; per_type = []
for nl in BUDGETS:
    for seed in SEEDS:
        te, tr, rng = split(seed)
        rng2 = np.random.RandomState(1000 + seed + nl)
        lab = draw_labels(rng2, tr, nl); yl = y[lab]
        scores = {'Pixel-diff thr.': b_pixel(lab, yl), 'SSIM thr.': b_ssim(lab, yl),
                  'kNN (labelled)': b_knn(lab, yl), 'Sup. GBM (labelled)': b_sup_gbm(lab, yl, seed),
                  'Sup. LR (labelled)': b_sup_lr(lab, yl), 'A-LP (proposed)': alp(lab, yl, tr, seed)}
        if nl == NL_MAIN:
            scores['Fully sup. GBM (all labels)'] = b_full(tr, seed)
        for m, s in scores.items():
            r = metrics(y[te], s[te], s[lab], yl); r.update(method=m, n_labels=nl, seed=seed); rows.append(r)
            if nl == NL_MAIN and seed == 0: pr_store[m] = precision_recall_curve(y[te], s[te])[:2]
        if nl == NL_MAIN:
            for m in ['Sup. LR (labelled)', 'A-LP (proposed)']:
                s = scores[m]; thr = np.quantile(s[te], 1 - max(yl.mean(), 1 / nl)); pred = s >= thr
                dd = d.iloc[te]
                for dc, g in dd.groupby('device_class'):
                    ii = te[(dd.device_class == dc).values]
                    per_dev.append(dict(method=m, seed=seed, device_class=dc, recall=float(pred[ii][y[ii] == 1].mean())))
                for dt, g in dd[dd.defect_type != 'none'].groupby('defect_type'):
                    ii = te[(dd.defect_type == dt).values]
                    per_type.append(dict(method=m, seed=seed, defect_type=dt, recall=float(pred[ii].mean())))
    print(f'budget {nl} done  t={time.time()-T0:.0f}s', flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, 'all_runs.csv'), index=False)
MK = ['pr_auc', 'roc_auc', 'f1', 'p_at_r90', 'fa_per_1k']
main = R[R.n_labels == NL_MAIN].groupby('method')[MK + ['prec', 'rec']].agg(['mean', 'std'])
main.columns = [f'{a}_{b}' for a, b in main.columns]
order = METHODS + ['Fully sup. GBM (all labels)']
main = main.loc[order]; main.round(4).to_csv(os.path.join(RES, 'table_main_comparison.csv'))
budget = R[R.method.isin(METHODS)].groupby(['method', 'n_labels'])[['pr_auc', 'f1']].agg(['mean', 'std'])
budget.columns = [f'{a}_{b}' for a, b in budget.columns]; budget.round(4).to_csv(os.path.join(RES, 'table_label_budget.csv'))
PD = pd.DataFrame(per_dev).groupby(['method', 'device_class']).recall.agg(['mean', 'std']).reset_index()
PD.round(4).to_csv(os.path.join(RES, 'table_recall_by_device.csv'), index=False)
PT = pd.DataFrame(per_type).groupby(['method', 'defect_type']).recall.agg(['mean', 'std']).reset_index()
PT.round(4).to_csv(os.path.join(RES, 'table_recall_by_defect_type.csv'), index=False)

# paired Wilcoxon over all (budget, seed) pairs: proposed vs each baseline (PR-AUC)
sig = {}
P = R[R.method == 'A-LP (proposed)'].set_index(['n_labels', 'seed']).pr_auc
for m in METHODS[:-1]:
    B = R[R.method == m].set_index(['n_labels', 'seed']).pr_auc.loc[P.index]
    st = wilcoxon(P.values, B.values, alternative='greater')
    sig[m] = dict(wins=int((P.values > B.values).sum()), n=int(len(P)), p_value=float(st.pvalue),
                  mean_diff=float((P.values - B.values).mean()))
pd.DataFrame(sig).T.to_csv(os.path.join(RES, 'table_significance.csv'))

# ---------------------------------------------------------------- 3. ablation (NL_MAIN, 5 seeds)
ABL = {'A-LP (full)': {}, 'w/o anchor prior (plain LP)': dict(use_prior=False),
       'w/o distillation': dict(use_distil=False), 'w/o rank fusion': dict(use_fuse=False),
       'Graph on all features': dict(space='all')}
arows = []
for seed in SEEDS:
    te, tr, rng = split(seed); lab = draw_labels(np.random.RandomState(1000 + seed + NL_MAIN), tr, NL_MAIN); yl = y[lab]
    for nm, kw in ABL.items():
        s = alp(lab, yl, tr, seed, **kw); r = metrics(y[te], s[te], s[lab], yl); r.update(variant=nm, seed=seed); arows.append(r)
A = pd.DataFrame(arows); abl = A.groupby('variant')[MK].agg(['mean', 'std']); abl.columns = [f'{a}_{b}' for a, b in abl.columns]
abl = abl.loc[list(ABL)]; abl.round(4).to_csv(os.path.join(RES, 'table_ablation.csv'))
print(f'ablation done t={time.time()-T0:.0f}s', flush=True)

# ---------------------------------------------------------------- 4. sensitivity: lambda x alpha (seeds 0-2)
LAMS = [0.5, 1, 2, 5, 10]; ALPHAS = [0.5, 0.8, 0.95]
sens = np.zeros((len(ALPHAS), len(LAMS)))
for seed in SEEDS[:3]:
    te, tr, rng = split(seed); lab = draw_labels(np.random.RandomState(1000 + seed + NL_MAIN), tr, NL_MAIN); yl = y[lab]
    pa = anchor(lab, yl); S = graph('evid', Xs[:, EI], CFG['k'])
    for i, a in enumerate(ALPHAS):
        for j, lm in enumerate(LAMS):
            ps = spread(S, lab, yl, pa, lm, a, CFG['iters'])
            s = 0.5 * rnk(distil(tr, lab, ps, seed)) + 0.5 * rnk(pa)
            sens[i, j] += AP(y[te], s[te]) / 3
pd.DataFrame(sens, index=[f'alpha={a}' for a in ALPHAS], columns=[f'lambda={l}' for l in LAMS]).round(4).to_csv(
    os.path.join(RES, 'table_sensitivity_lambda_alpha.csv'))
print(f'sensitivity done t={time.time()-T0:.0f}s', flush=True)

# ---------------------------------------------------------------- 5. cross-device generalisation
crow = []
phones = ~d.device_class.isin(['foldable_android', 'tablet']).values
for seed in SEEDS:
    te, tr, rng = split(seed)
    tr_ph = tr[phones[tr]]; te_big = te[~phones[te]]; te_ph = te[phones[te]]
    lab = draw_labels(np.random.RandomState(2000 + seed), tr_ph, NL_MAIN); yl = y[lab]
    sc = {'Sup. GBM (labelled)': b_sup_gbm(lab, yl, seed), 'Sup. LR (labelled)': b_sup_lr(lab, yl),
          'A-LP (proposed)': alp(lab, yl, tr, seed)}   # unlabelled pool includes foldables/tablets
    for m, s in sc.items():
        for nm, tt in [('phones', te_ph), ('foldables+tablets', te_big)]:
            r = metrics(y[tt], s[tt], s[lab], yl); r.update(method=m, seed=seed, test=nm); crow.append(r)
CD = pd.DataFrame(crow); cd = CD.groupby(['method', 'test'])[['pr_auc', 'f1']].agg(['mean', 'std'])
cd.columns = [f'{a}_{b}' for a, b in cd.columns]; cd.round(4).to_csv(os.path.join(RES, 'table_cross_device.csv'))
print(f'cross-device done t={time.time()-T0:.0f}s', flush=True)

# convergence trace of Eq. (5)
tr_c = []; te, tr, rng = split(0); lab = draw_labels(np.random.RandomState(1000 + NL_MAIN), tr, NL_MAIN)
alp(lab, y[lab], tr, 0, trace=tr_c)

# ---------------------------------------------------------------- 6. figures
COL = {'Pixel-diff thr.': '#9e9e9e', 'SSIM thr.': '#bdbdbd', 'kNN (labelled)': '#7f7f7f',
       'Sup. GBM (labelled)': '#5b8db8', 'Sup. LR (labelled)': '#2c5d8a', 'A-LP (proposed)': '#d1495b',
       'Fully sup. GBM (all labels)': '#000000'}
def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(RES, name + '.jpg'), dpi=300, bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.4))
for m in METHODS:
    b = budget.loc[m]; ax.errorbar(BUDGETS, b['pr_auc_mean'], yerr=b['pr_auc_std'], marker='o', ms=3, capsize=2,
                                   lw=1.6 if 'A-LP' in m else 1, color=COL[m], label=m)
fu = main.loc['Fully sup. GBM (all labels)', 'pr_auc_mean']
ax.axhline(fu, ls='--', color='k', lw=0.8, label='Fully sup. GBM (12,800 labels)')
ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
ax.set_xlabel('Number of labelled comparisons $n_L$'); ax.set_ylabel('PR-AUC (test)'); ax.set_ylim(0, 1)
ax.legend(ncol=2, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False); ax.grid(alpha=.3)
save(fig, 'fig2_label_budget')

fig, ax = plt.subplots(figsize=(3.4, 2.4))
for m in order:
    p, r = pr_store[m]; ax.plot(r, p, color=COL[m], lw=1.6 if 'A-LP' in m else 1,
                                ls='--' if 'Fully' in m else '-', label=m)
ax.axvline(0.9, color='grey', lw=0.6, ls=':')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
ax.legend(fontsize=6, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False)
save(fig, 'fig3_pr_curves')

fig, ax = plt.subplots(figsize=(3.4, 2.0))
im = ax.imshow(sens, cmap='Reds', aspect='auto')
for i in range(len(ALPHAS)):
    for j in range(len(LAMS)):
        ax.text(j, i, f'{sens[i, j]:.3f}', ha='center', va='center', fontsize=7,
                color='white' if sens[i, j] > sens.mean() + 0.3 * (sens.max() - sens.mean()) else 'black')
ax.set_xticks(range(len(LAMS))); ax.set_xticklabels(LAMS); ax.set_yticks(range(len(ALPHAS))); ax.set_yticklabels(ALPHAS)
ax.set_xlabel(r'Anchor weight $\lambda$'); ax.set_ylabel(r'Propagation $\alpha$')
fig.colorbar(im, ax=ax, label='PR-AUC')
save(fig, 'fig4_sensitivity')

fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.5))
for ax, (tab, key, ttl) in zip(axs, [(PD, 'device_class', 'Device class'), (PT, 'defect_type', 'Defect type')]):
    cats = sorted(tab[key].unique()); xx = np.arange(len(cats)); wdt = 0.4
    for o, m in [(-wdt / 2, 'Sup. LR (labelled)'), (wdt / 2, 'A-LP (proposed)')]:
        t = tab[tab.method == m].set_index(key).loc[cats]
        ax.barh(xx + o, t['mean'], wdt, xerr=t['std'], color=COL[m], label=m.split(' (')[0], error_kw=dict(lw=0.6))
    ax.set_yticks(xx); ax.set_yticklabels([c.replace('_', ' ') for c in cats], fontsize=6.5)
    ax.set_xlabel('Recall'); ax.set_title(ttl); ax.set_xlim(0, 1); ax.grid(alpha=.3, axis='x')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=6.5, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(); save(fig, 'fig5_recall_breakdown')

fig, ax = plt.subplots(figsize=(3.4, 1.8))
ax.semilogy(range(1, len(tr_c) + 1), tr_c, color=COL['A-LP (proposed)'], marker='.', ms=3)
ax.set_xlabel('Iteration t of Eq. (5)'); ax.set_ylabel(r'Rel. change $\|F^{t+1}-F^{t}\|_1/\|F^{t}\|_1$'); ax.grid(alpha=.3)
save(fig, 'fig6_convergence')

# ---------------------------------------------------------------- 7. dataset summary + metrics.json
summ = {'rows': N, 'columns': d.shape[1], 'defect_rate': float(y.mean()), 'screens': int(d.screen_id.nunique()),
        'period': f'{d.build_ts.min()[:10]} to {d.build_ts.max()[:10]}',
        'defect_types': d.defect_type.value_counts().to_dict(), 'device_classes': d.device_class.value_counts().to_dict(),
        'missing_rate': {k: v for k, v in miss.items() if v > 0}, 'winsorised_values': clip_counts,
        'features_used': len(COLS), 'evidence_features': EVID,
        'defect_rate_by_device': d.groupby('device_class').is_ui_defect.mean().round(4).to_dict(),
        'pixel_diff_pct_mean_defect_vs_ok': [float(d.pixel_diff_pct[y == 1].mean()), float(d.pixel_diff_pct[y == 0].mean())]}
pd.Series({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in summ.items()}).to_csv(os.path.join(RES, 'table_dataset_summary.csv'))
out = dict(config=CFG, n_labels_main=NL_MAIN, seeds=SEEDS, n_test=N_TEST, dataset=summ,
           main={m: {c: float(main.loc[m, c]) for c in main.columns} for m in order},
           ablation={v: {c: float(abl.loc[v, c]) for c in abl.columns} for v in abl.index},
           significance=sig, sensitivity=sens.round(4).tolist(), sens_lambdas=LAMS, sens_alphas=ALPHAS,
           cross_device={f'{a}|{b}': {c: float(cd.loc[(a, b), c]) for c in cd.columns} for a, b in cd.index},
           budget={f'{a}|{b}': {c: float(budget.loc[(a, b), c]) for c in budget.columns} for a, b in budget.index},
           convergence_trace=tr_c, runtime_s=time.time() - T0)
json.dump(out, open(os.path.join(RES, 'metrics.json'), 'w'), indent=1)
print(main[['pr_auc_mean', 'pr_auc_std', 'f1_mean', 'p_at_r90_mean', 'fa_per_1k_mean']].round(3))
print(abl[['pr_auc_mean', 'pr_auc_std', 'f1_mean']].round(3)); print(cd.round(3)); print(sig)
print(f'TOTAL runtime {time.time()-T0:.0f}s')
