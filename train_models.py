"""
train_models.py
Customer Churn Prediction — Full ML Pipeline
Models: Logistic Regression, Random Forest, Gradient Boosting (XGBoost-equivalent)
Outputs: trained models, evaluation charts, metrics JSON
"""

import pandas as pd
import numpy as np
import json, pickle, warnings, os
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import OneHotEncoder
from sklearn.impute            import SimpleImputer
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection        import permutation_importance
from sklearn.calibration        import calibration_curve

# ── Theme ────────────────────────────────────────────────────
BG, BG2     = '#0f1117', '#1a1d27'
BORDER      = '#2e3250'
GREEN       = '#4ade80'
RED         = '#f87171'
BLUE        = '#60a5fa'
YELLOW      = '#fbbf24'
PURPLE      = '#a78bfa'
ORANGE      = '#fb923c'
MUTED       = '#8892b0'
TEXT        = '#e2e8f0'
ACCENT      = '#38bdf8'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG2,
    'axes.edgecolor': BORDER, 'axes.labelcolor': TEXT,
    'text.color': TEXT, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'grid.color': BORDER, 'grid.alpha': 0.35, 'font.family': 'monospace',
    'axes.spines.top': False, 'axes.spines.right': False,
})

OUT = '/home/claude/churn_project/outputs/'
MDL = '/home/claude/churn_project/models/'
os.makedirs(OUT, exist_ok=True)
os.makedirs(MDL, exist_ok=True)

# ════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS
# ════════════════════════════════════════════════════════════
df = pd.read_csv('/home/claude/churn_project/data/telecom_churn.csv')
df = df.drop(columns=['CustomerID'])
print(f"Loaded {len(df):,} rows  |  Churn={df['Churn'].mean():.1%}")

# Separate features / target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Column types
num_cols = ['TenureMonths','MonthlyCharges','TotalCharges']
cat_cols = X.select_dtypes('object').columns.tolist()

# ── Preprocessing pipelines ──────────────────────────────────
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols),
])

# ── Train / test split (stratified) ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Fit preprocessor ────────────────────────────────────────
X_train_t = preprocessor.fit_transform(X_train)
X_test_t  = preprocessor.transform(X_test)

# Feature names after OHE
ohe_names   = preprocessor.named_transformers_['cat']['ohe'].get_feature_names_out(cat_cols)
feature_names = np.array(num_cols + list(ohe_names))
print(f"Feature dimensions: {X_train_t.shape[1]}")

# ════════════════════════════════════════════════════════════
# 2. TRAIN MODELS
# ════════════════════════════════════════════════════════════
models = {
    'Logistic Regression': LogisticRegression(
        C=1.0, max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        random_state=42, class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42),
}

results = {}
trained = {}
skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_t, y_train)
    trained[name] = model

    y_pred      = model.predict(X_test_t)
    y_prob      = model.predict_proba(X_test_t)[:, 1]

    # CV ROC-AUC
    cv_auc = cross_val_score(model, X_train_t, y_train,
                              cv=skf, scoring='roc_auc', n_jobs=-1)

    results[name] = {
        'accuracy':   accuracy_score(y_test, y_pred),
        'precision':  precision_score(y_test, y_pred),
        'recall':     recall_score(y_test, y_pred),
        'f1':         f1_score(y_test, y_pred),
        'roc_auc':    roc_auc_score(y_test, y_prob),
        'avg_precision': average_precision_score(y_test, y_prob),
        'cv_auc_mean':cv_auc.mean(),
        'cv_auc_std': cv_auc.std(),
        'y_prob':     y_prob,
        'y_pred':     y_pred,
    }

    print(f"  Accuracy:  {results[name]['accuracy']:.4f}")
    print(f"  Recall:    {results[name]['recall']:.4f}")
    print(f"  ROC-AUC:   {results[name]['roc_auc']:.4f}  |  CV: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# Save models
for name, model in trained.items():
    safe = name.lower().replace(' ', '_')
    with open(f'{MDL}{safe}.pkl', 'wb') as f:
        pickle.dump({'model': model, 'preprocessor': preprocessor,
                     'feature_names': feature_names}, f)
print(f"\nModels saved to {MDL}")

# ════════════════════════════════════════════════════════════
# 3. EVALUATION CHARTS
# ════════════════════════════════════════════════════════════
MODEL_COLORS = {
    'Logistic Regression': BLUE,
    'Random Forest':       GREEN,
    'Gradient Boosting':   YELLOW,
}

# ── Chart 9: Metrics Comparison ─────────────────────────────
metrics_to_plot = ['accuracy','precision','recall','f1','roc_auc']
metric_labels   = ['Accuracy','Precision','Recall','F1','ROC-AUC']

fig, axes = plt.subplots(1, 5, figsize=(18, 6), facecolor=BG)
fig.suptitle('Model Performance — Test Set Metrics Comparison',
             fontsize=14, color=TEXT, fontweight='bold', y=1.02)

for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
    ax = axes[i]
    vals   = [results[m][metric] for m in models.keys()]
    colors = [MODEL_COLORS[m] for m in models.keys()]
    bars   = ax.bar(range(3), vals, color=colors, edgecolor=BG,
                    linewidth=0.8, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9,
                color=TEXT, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['LR', 'RF', 'GB'], fontsize=9)
    ax.set_title(label, color=ACCENT, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.12)
    ax.set_facecolor(BG2); ax.grid(axis='y', alpha=0.25)
    ax.tick_params(colors=MUTED)
    if i == 0:
        patches = [mpatches.Patch(color=c, label=n)
                   for n, c in MODEL_COLORS.items()]
        ax.legend(handles=patches, fontsize=7, framealpha=0,
                  loc='lower right')

plt.tight_layout()
plt.savefig(OUT + 'model_09_metrics_comparison.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 9: Metrics comparison")

# ── Chart 10: ROC Curves (all 3) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    auc = res['roc_auc']
    axes[0].plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2.2,
                 label=f"{name}  (AUC={auc:.3f})")

axes[0].plot([0,1],[0,1], color=MUTED, linestyle='--', linewidth=1, alpha=0.5)
axes[0].fill_between([0,1],[0,1],[0,1], alpha=0.04, color=MUTED)
axes[0].set_xlabel('False Positive Rate', color=MUTED, fontsize=11)
axes[0].set_ylabel('True Positive Rate', color=MUTED, fontsize=11)
axes[0].set_title('ROC Curves — All Models', color=TEXT, fontsize=13)
axes[0].legend(fontsize=8.5, framealpha=0.1,
               facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT)
axes[0].set_facecolor(BG2); axes[0].grid(alpha=0.2)

# Precision-Recall curves
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ap = res['avg_precision']
    axes[1].plot(rec, prec, color=MODEL_COLORS[name], linewidth=2.2,
                 label=f"{name}  (AP={ap:.3f})")

baseline = y_test.mean()
axes[1].axhline(baseline, color=MUTED, linestyle='--', linewidth=1, alpha=0.5,
                label=f"Baseline ({baseline:.2f})")
axes[1].set_xlabel('Recall', color=MUTED, fontsize=11)
axes[1].set_ylabel('Precision', color=MUTED, fontsize=11)
axes[1].set_title('Precision-Recall Curves', color=TEXT, fontsize=13)
axes[1].legend(fontsize=8.5, framealpha=0.1,
               facecolor=BG2, edgecolor=BORDER, labelcolor=TEXT)
axes[1].set_facecolor(BG2); axes[1].grid(alpha=0.2)

for ax in axes:
    ax.tick_params(colors=MUTED)
    ax.spines[:].set_color(BORDER)

plt.tight_layout()
plt.savefig(OUT + 'model_10_roc_pr_curves.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 10: ROC + PR curves")

# ── Chart 11: Confusion Matrices (all 3) ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
fig.suptitle('Confusion Matrices — Test Set', fontsize=14,
             color=TEXT, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    cm     = confusion_matrix(y_test, res['y_pred'])
    cm_pct = cm.astype(float) / cm.sum() * 100

    im = ax.imshow(cm, cmap='Blues', aspect='auto', interpolation='nearest')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nRetained', 'Predicted\nChurned'],
                       fontsize=9, color=TEXT)
    ax.set_yticklabels(['Actual\nRetained', 'Actual\nChurned'],
                       fontsize=9, color=TEXT)

    cell_colors = [['white','#f87171'],['#86efac','white']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color=BG2)
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                fill=True, color=cell_colors[i][j], alpha=0.55, zorder=0))

    auc_str = f"ROC-AUC: {res['roc_auc']:.3f}"
    ax.set_title(f"{name}\n{auc_str}", color=ACCENT, fontsize=10,
                 fontweight='bold', pad=10)
    ax.set_facecolor(BG2)
    ax.spines[:].set_color(BORDER)

plt.tight_layout()
plt.savefig(OUT + 'model_11_confusion_matrices.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 11: Confusion matrices")

# ── Chart 12: Feature Importance (Random Forest + GBM) ──────
fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor=BG)
fig.suptitle('Feature Importance Analysis', fontsize=14,
             color=TEXT, fontweight='bold')

for ax, (model_name, importance_attr, color) in zip(axes, [
    ('Random Forest',    'feature_importances_', GREEN),
    ('Gradient Boosting','feature_importances_', YELLOW),
]):
    imp   = getattr(trained[model_name], importance_attr)
    top_n = 20
    top_idx = np.argsort(imp)[-top_n:]
    top_names = feature_names[top_idx]
    top_imp   = imp[top_idx]

    # Shorten feature names
    short = []
    for n in top_names:
        n = (n.replace('PaymentMethod_','PM: ')
              .replace('InternetService_','IS: ')
              .replace('Contract_','Ctr: ')
              .replace('OnlineSecurity_','OSecure: ')
              .replace('TechSupport_','TechSup: ')
              .replace('MultipleLines_','ML: '))
        short.append(n[:30])

    bar_colors = [color if v > np.percentile(top_imp, 75) else
                  BLUE   if v > np.percentile(top_imp, 50) else
                  MUTED  for v in top_imp]

    bars = ax.barh(range(top_n), top_imp, color=bar_colors,
                   edgecolor=BG, linewidth=0.5, height=0.72)
    for b, v in zip(bars, top_imp):
        ax.text(v + 0.0003, b.get_y() + b.get_height()/2,
                f'{v:.4f}', va='center', fontsize=7, color=TEXT)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(short, fontsize=7.5)
    ax.set_xlabel('Importance Score', color=MUTED, fontsize=10)
    ax.set_title(f'{model_name}\nTop-20 Feature Importances',
                 color=ACCENT, fontsize=11, fontweight='bold')
    ax.set_facecolor(BG2); ax.grid(axis='x', alpha=0.25)
    ax.tick_params(colors=MUTED)

plt.tight_layout()
plt.savefig(OUT + 'model_12_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 12: Feature importance")

# ── Chart 13: Cross-validation + Calibration ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle('Model Reliability — CV Scores & Calibration', 
             fontsize=14, color=TEXT, fontweight='bold')

# CV AUC distributions (simulated per-fold)
cv_scores_all = {}
for name, model in trained.items():
    cv_s = cross_val_score(model, X_train_t, y_train,
                            cv=skf, scoring='roc_auc', n_jobs=-1)
    cv_scores_all[name] = cv_s

pos = list(range(len(models)))
for i, (name, scores) in enumerate(cv_scores_all.items()):
    bp = axes[0].boxplot(scores, positions=[i], widths=0.4,
        patch_artist=True, notch=False, showfliers=True,
        boxprops=dict(facecolor=MODEL_COLORS[name], alpha=0.6, color=TEXT),
        whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED),
        medianprops=dict(color=YELLOW, linewidth=2.5),
        flierprops=dict(marker='o', color=MUTED, markersize=4))
    axes[0].text(i, scores.mean() + 0.002, f'{scores.mean():.4f}',
                 ha='center', va='bottom', fontsize=8, color=TEXT,
                 fontweight='bold')

axes[0].set_xticks(pos)
axes[0].set_xticklabels(['LR', 'RF', 'GB'], fontsize=10)
axes[0].set_ylabel('ROC-AUC (5-fold CV)', color=MUTED, fontsize=10)
axes[0].set_title('Cross-Validation ROC-AUC Distribution', color=TEXT, fontsize=11)
axes[0].set_facecolor(BG2); axes[0].grid(axis='y', alpha=0.25)

# Calibration curves
axes[1].plot([0,1],[0,1], color=MUTED, linestyle='--', linewidth=1.2,
             label='Perfectly calibrated', alpha=0.6)
for name, res in results.items():
    prob_true, prob_pred = calibration_curve(y_test, res['y_prob'], n_bins=10)
    axes[1].plot(prob_pred, prob_true, color=MODEL_COLORS[name],
                 marker='o', markersize=5, linewidth=2, label=name)

axes[1].set_xlabel('Mean Predicted Probability', color=MUTED, fontsize=10)
axes[1].set_ylabel('Fraction of Positives', color=MUTED, fontsize=10)
axes[1].set_title('Calibration Curves (Reliability Diagram)', color=TEXT, fontsize=11)
axes[1].legend(fontsize=8.5, framealpha=0.1, facecolor=BG2,
               edgecolor=BORDER, labelcolor=TEXT)
axes[1].set_facecolor(BG2); axes[1].grid(alpha=0.2)

for ax in axes:
    ax.tick_params(colors=MUTED); ax.spines[:].set_color(BORDER)

plt.tight_layout()
plt.savefig(OUT + 'model_13_cv_calibration.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 13: CV + Calibration")

# ── Chart 14: Threshold Tuning (Gradient Boosting) ──────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.suptitle('Gradient Boosting — Threshold Optimisation',
             fontsize=14, color=TEXT, fontweight='bold')

gb_prob = results['Gradient Boosting']['y_prob']
thresholds = np.linspace(0.01, 0.99, 200)
precisions, recalls, f1s, accs = [], [], [], []

for t in thresholds:
    preds = (gb_prob >= t).astype(int)
    precisions.append(precision_score(y_test, preds, zero_division=0))
    recalls.append(recall_score(y_test, preds, zero_division=0))
    f1s.append(f1_score(y_test, preds, zero_division=0))
    accs.append(accuracy_score(y_test, preds))

best_f1_idx   = np.argmax(f1s)
best_threshold = thresholds[best_f1_idx]

axes[0].plot(thresholds, precisions, color=BLUE,   linewidth=2, label='Precision')
axes[0].plot(thresholds, recalls,    color=RED,    linewidth=2, label='Recall')
axes[0].plot(thresholds, f1s,        color=YELLOW, linewidth=2.5, label='F1 Score')
axes[0].plot(thresholds, accs,       color=GREEN,  linewidth=1.5,
             linestyle='--', label='Accuracy', alpha=0.7)
axes[0].axvline(best_threshold, color=PURPLE, linestyle=':', linewidth=2,
                label=f'Best F1 threshold={best_threshold:.2f}')
axes[0].axvline(0.5, color=MUTED, linestyle='--', linewidth=1, alpha=0.5,
                label='Default threshold=0.5')
axes[0].fill_betweenx([0,1], best_threshold-0.01, best_threshold+0.01,
                       alpha=0.15, color=PURPLE)
axes[0].set_xlabel('Decision Threshold', color=MUTED, fontsize=10)
axes[0].set_ylabel('Score', color=MUTED, fontsize=10)
axes[0].set_title('Metrics vs Decision Threshold', color=TEXT, fontsize=11)
axes[0].legend(fontsize=8, framealpha=0.1, facecolor=BG2,
               edgecolor=BORDER, labelcolor=TEXT)
axes[0].set_facecolor(BG2); axes[0].grid(alpha=0.2)

# Business impact: FP vs FN cost simulation
# Assume: missing a churner costs $300, false alarm costs $30
churn_cost, false_alarm_cost = 300, 30
total_costs = []
for t in thresholds:
    preds = (gb_prob >= t).astype(int)
    cm    = confusion_matrix(y_test, preds)
    fn    = cm[1, 0] if cm.shape == (2,2) else 0
    fp    = cm[0, 1] if cm.shape == (2,2) else 0
    total_costs.append(fn * churn_cost + fp * false_alarm_cost)

best_cost_idx = np.argmin(total_costs)
axes[1].plot(thresholds, total_costs, color=ORANGE, linewidth=2.5)
axes[1].axvline(thresholds[best_cost_idx], color=GREEN, linestyle=':',
                linewidth=2, label=f'Min cost threshold={thresholds[best_cost_idx]:.2f}')
axes[1].axvline(0.5, color=MUTED, linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_xlabel('Decision Threshold', color=MUTED, fontsize=10)
axes[1].set_ylabel('Total Business Cost ($)', color=MUTED, fontsize=10)
axes[1].set_title('Business Cost vs Threshold\n(FN=$300 · FP=$30)',
                  color=TEXT, fontsize=11)
axes[1].legend(fontsize=9, framealpha=0.1, facecolor=BG2,
               edgecolor=BORDER, labelcolor=TEXT)
axes[1].set_facecolor(BG2); axes[1].grid(alpha=0.2)
axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'${x:,.0f}'))

for ax in axes:
    ax.tick_params(colors=MUTED); ax.spines[:].set_color(BORDER)

plt.tight_layout()
plt.savefig(OUT + 'model_14_threshold_tuning.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 14: Threshold tuning")

# ── Chart 15: Churn Probability Score Distribution ───────────
fig, axes = plt.subplots(1, 3, figsize=(17, 6), facecolor=BG)
fig.suptitle('Predicted Churn Probability — Score Distributions',
             fontsize=14, color=TEXT, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    prob = res['y_prob']
    for churn_val, color, lbl in [(0, BLUE, 'Retained'), (1, RED, 'Churned')]:
        mask = y_test == churn_val
        ax.hist(prob[mask], bins=40, density=True, alpha=0.55,
                color=color, label=lbl, edgecolor='none')
    ax.axvline(0.5, color=YELLOW, linestyle='--', linewidth=1.5,
               label='Default threshold', alpha=0.8)
    ax.set_title(name, color=ACCENT, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Churn Probability', color=MUTED, fontsize=9)
    ax.set_ylabel('Density', color=MUTED, fontsize=9)
    ax.legend(fontsize=8, framealpha=0)
    ax.set_facecolor(BG2); ax.grid(alpha=0.2)
    ax.tick_params(colors=MUTED); ax.spines[:].set_color(BORDER)
    # AUC annotation
    ax.text(0.97, 0.95, f"AUC={res['roc_auc']:.3f}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, color=YELLOW, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG2,
                      edgecolor=BORDER, alpha=0.8))

plt.tight_layout()
plt.savefig(OUT + 'model_15_score_distributions.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 15: Score distributions")

# ════════════════════════════════════════════════════════════
# 4. FINAL METRICS TABLE + JSON REPORT
# ════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("FINAL MODEL EVALUATION REPORT")
print("═"*70)

header = f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9} {'CV-AUC':>12}"
print(header)
print("─"*70)

report_data = {}
for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:>9.4f} {res['precision']:>10.4f} "
          f"{res['recall']:>8.4f} {res['f1']:>8.4f} "
          f"{res['roc_auc']:>9.4f} "
          f"{res['cv_auc_mean']:>6.4f}±{res['cv_auc_std']:.4f}")
    report_data[name] = {k: round(float(v), 4)
                         for k, v in res.items()
                         if k not in ('y_prob','y_pred')}

print("─"*70)

# Best model
best_model = max(results, key=lambda m: results[m]['roc_auc'])
print(f"\n🏆  Best Model: {best_model}  "
      f"(ROC-AUC = {results[best_model]['roc_auc']:.4f})")
print(f"    Best F1 Threshold for GB: {best_threshold:.3f}")
print(f"    Min-cost Threshold for GB: {thresholds[best_cost_idx]:.3f}")

# Save JSON report
report = {
    'dataset':   {'rows': len(df), 'churn_rate': round(df['Churn'].mean(), 4),
                  'train_size': len(X_train), 'test_size': len(X_test)},
    'models':    report_data,
    'best_model':best_model,
    'thresholds':{
        'default': 0.5,
        'best_f1': round(best_threshold, 3),
        'min_cost':round(thresholds[best_cost_idx], 3),
    },
}
with open(OUT + 'evaluation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"\nAll outputs saved to {OUT}")
print("Models saved to    " + MDL)
print("\n✓ Pipeline complete — 7 evaluation charts generated")
