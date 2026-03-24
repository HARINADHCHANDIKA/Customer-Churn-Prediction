"""
eda.py
Exploratory Data Analysis for Customer Churn Prediction
Generates 8 charts saved to /outputs/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Style ────────────────────────────────────────────────────
BG      = '#0f1117'
BG2     = '#1a1d27'
PANEL   = '#22263a'
BORDER  = '#2e3250'
GREEN   = '#4ade80'
RED     = '#f87171'
BLUE    = '#60a5fa'
YELLOW  = '#fbbf24'
PURPLE  = '#a78bfa'
MUTED   = '#8892b0'
TEXT    = '#e2e8f0'
ACCENT  = '#38bdf8'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    BG2,
    'axes.edgecolor':    BORDER,
    'axes.labelcolor':   TEXT,
    'text.color':        TEXT,
    'xtick.color':       MUTED,
    'ytick.color':       MUTED,
    'grid.color':        BORDER,
    'grid.alpha':        0.4,
    'font.family':       'monospace',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

CHURN_COLORS = {0: BLUE, 1: RED}
CHURN_LABELS = {0: 'Retained', 1: 'Churned'}

df = pd.read_csv('/home/claude/churn_project/data/telecom_churn.csv')
print(f"Loaded {len(df):,} rows  |  Churn rate: {df['Churn'].mean():.1%}")

OUT = '/home/claude/churn_project/outputs/'

# ════════════════════════════════════════════════════════════
# CHART 1 — Overview Dashboard (2×3 grid)
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 11), facecolor=BG)
fig.suptitle('Customer Churn — EDA Overview Dashboard', fontsize=16,
             fontweight='bold', color=TEXT, y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# 1a — Churn distribution (donut)
ax1 = fig.add_subplot(gs[0, 0])
churn_counts = df['Churn'].value_counts().sort_index()
wedge_props  = dict(width=0.45, edgecolor=BG, linewidth=2)
wedges, texts, autotexts = ax1.pie(
    churn_counts, labels=['Retained', 'Churned'],
    colors=[BLUE, RED], autopct='%1.1f%%',
    startangle=90, wedgeprops=wedge_props,
    textprops={'color': TEXT, 'fontsize': 9})
for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
ax1.set_title('Churn Distribution', color=TEXT, fontsize=11, pad=10)
ax1.text(0, 0, f'n={len(df):,}', ha='center', va='center',
         color=MUTED, fontsize=8)

# 1b — Churn by Contract type
ax2 = fig.add_subplot(gs[0, 1])
ct = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
bars = ax2.bar(ct.index, ct.values * 100, color=[RED, YELLOW, GREEN],
               edgecolor=BG, linewidth=0.8, width=0.55)
for b, v in zip(bars, ct.values):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
             f'{v:.1%}', ha='center', va='bottom', fontsize=9,
             color=TEXT, fontweight='bold')
ax2.set_title('Churn Rate by Contract', color=TEXT, fontsize=11)
ax2.set_ylabel('Churn Rate (%)', color=MUTED, fontsize=9)
ax2.set_xticklabels(ct.index, fontsize=8, rotation=12)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax2.grid(axis='y', alpha=0.3)

# 1c — Churn by Internet Service
ax3 = fig.add_subplot(gs[0, 2])
it = df.groupby('InternetService')['Churn'].mean().sort_values(ascending=False)
colors_is = [RED if x == max(it.values) else BLUE for x in it.values]
bars3 = ax3.bar(it.index, it.values * 100, color=colors_is,
                edgecolor=BG, linewidth=0.8, width=0.55)
for b, v in zip(bars3, it.values):
    ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
             f'{v:.1%}', ha='center', va='bottom', fontsize=9,
             color=TEXT, fontweight='bold')
ax3.set_title('Churn Rate by Internet Service', color=TEXT, fontsize=11)
ax3.set_ylabel('Churn Rate (%)', color=MUTED, fontsize=9)
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax3.grid(axis='y', alpha=0.3)

# 1d — Tenure distribution (KDE by churn)
ax4 = fig.add_subplot(gs[1, 0])
for churn_val, color, lbl in [(0, BLUE, 'Retained'), (1, RED, 'Churned')]:
    data = df[df['Churn'] == churn_val]['TenureMonths']
    ax4.hist(data, bins=30, alpha=0.55, color=color, label=lbl,
             density=True, edgecolor='none')
ax4.set_title('Tenure Distribution by Churn', color=TEXT, fontsize=11)
ax4.set_xlabel('Tenure (months)', color=MUTED, fontsize=9)
ax4.set_ylabel('Density', color=MUTED, fontsize=9)
ax4.legend(fontsize=8, framealpha=0)

# 1e — Monthly charges violin
ax5 = fig.add_subplot(gs[1, 1])
parts = ax5.violinplot(
    [df[df['Churn']==0]['MonthlyCharges'].dropna(),
     df[df['Churn']==1]['MonthlyCharges'].dropna()],
    positions=[0, 1], showmedians=True, showmeans=False)
for pc, color in zip(parts['bodies'], [BLUE, RED]):
    pc.set_facecolor(color); pc.set_alpha(0.6)
parts['cmedians'].set_color(YELLOW)
parts['cmedians'].set_linewidth(2)
for key in ['cbars','cmins','cmaxes']:
    parts[key].set_color(MUTED); parts[key].set_linewidth(1)
ax5.set_xticks([0, 1]); ax5.set_xticklabels(['Retained', 'Churned'], fontsize=9)
ax5.set_title('Monthly Charges Distribution', color=TEXT, fontsize=11)
ax5.set_ylabel('Monthly Charges ($)', color=MUTED, fontsize=9)
ax5.grid(axis='y', alpha=0.3)

# 1f — Payment method churn rate
ax6 = fig.add_subplot(gs[1, 2])
pm = df.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=True)
short_labels = [p.replace(' (automatic)', '\n(auto)').replace(' check', '\ncheck') for p in pm.index]
bars6 = ax6.barh(range(len(pm)), pm.values * 100,
                 color=[RED if v == max(pm.values) else BLUE for v in pm.values],
                 edgecolor=BG, linewidth=0.8, height=0.55)
for i, (b, v) in enumerate(zip(bars6, pm.values)):
    ax6.text(v + 0.3, b.get_y() + b.get_height()/2,
             f'{v:.1%}', va='center', fontsize=8, color=TEXT)
ax6.set_yticks(range(len(pm))); ax6.set_yticklabels(short_labels, fontsize=7)
ax6.set_title('Churn Rate by Payment Method', color=TEXT, fontsize=11)
ax6.set_xlabel('Churn Rate (%)', color=MUTED, fontsize=9)
ax6.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax6.grid(axis='x', alpha=0.3)

plt.savefig(OUT + 'eda_01_overview_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 1: Overview dashboard")

# ════════════════════════════════════════════════════════════
# CHART 2 — Correlation Heatmap (encoded features)
# ════════════════════════════════════════════════════════════
binary_cols = ['Gender','Partner','Dependents','PhoneService','PaperlessBilling','Churn']
ordinal_map = {'No phone service':0,'No':1,'Yes':2,
               'No internet service':0,'DSL':1,'Fiber optic':2,
               'Month-to-month':0,'One year':1,'Two year':2,
               'Male':0,'Female':1}

df_enc = df.copy()
cat_cols = df.select_dtypes('object').columns.drop('CustomerID')
for col in cat_cols:
    df_enc[col] = df_enc[col].map(ordinal_map).fillna(df_enc[col].astype('category').cat.codes)

numeric_cols = ['TenureMonths','MonthlyCharges','TotalCharges','SeniorCitizen',
                'Contract','InternetService','OnlineSecurity','TechSupport',
                'PaperlessBilling','PaymentMethod','Churn']
corr = df_enc[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.8, vmin=-0.8,
            center=0, annot=True, fmt='.2f', annot_kws={'size': 8},
            linewidths=0.5, linecolor=BG, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_facecolor(BG2)
ax.set_title('Feature Correlation Matrix\n(Encoded Variables)', color=TEXT,
             fontsize=14, pad=15)
ax.tick_params(labelsize=8, colors=MUTED)
plt.savefig(OUT + 'eda_02_correlation_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 2: Correlation heatmap")

# ════════════════════════════════════════════════════════════
# CHART 3 — Churn rate by every categorical feature
# ════════════════════════════════════════════════════════════
cat_features = ['Contract','InternetService','PaymentMethod','OnlineSecurity',
                'TechSupport','PaperlessBilling','Partner','Dependents',
                'SeniorCitizen','MultipleLines']

fig, axes = plt.subplots(2, 5, figsize=(22, 9), facecolor=BG)
fig.suptitle('Churn Rate Across Categorical Features', fontsize=15,
             color=TEXT, fontweight='bold', y=1.01)
axes = axes.flatten()

for i, feat in enumerate(cat_features):
    ax = axes[i]
    grp = df.groupby(feat)['Churn'].agg(['mean','count']).reset_index()
    grp = grp.sort_values('mean', ascending=False)
    palette = [RED if v == grp['mean'].max() else
               GREEN if v == grp['mean'].min() else BLUE
               for v in grp['mean']]
    bars = ax.bar(range(len(grp)), grp['mean'] * 100,
                  color=palette, edgecolor=BG, linewidth=0.5, width=0.6)
    for j, (b, row) in enumerate(zip(bars, grp.itertuples())):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f'{row.mean:.0%}', ha='center', va='bottom',
                fontsize=7, color=TEXT)
        ax.text(b.get_x() + b.get_width()/2, b.get_height()/2,
                f'n={row.count:,}', ha='center', va='center',
                fontsize=6, color=BG, fontweight='bold')
    ax.set_title(feat, color=ACCENT, fontsize=9, fontweight='bold')
    labels = [str(v)[:12] for v in grp[feat]]
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=20, ha='right')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.set_facecolor(BG2); ax.grid(axis='y', alpha=0.25)
    ax.tick_params(colors=MUTED, labelsize=7)

plt.tight_layout()
plt.savefig(OUT + 'eda_03_categorical_churn_rates.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 3: Categorical churn rates")

# ════════════════════════════════════════════════════════════
# CHART 4 — Tenure vs Monthly Charges scatter (churn coloured)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=BG)
fig.suptitle('Customer Segments — Tenure × Monthly Charges',
             fontsize=14, color=TEXT, fontweight='bold')

for churn_val, color, label in [(0, BLUE, 'Retained'), (1, RED, 'Churned')]:
    sub = df[df['Churn'] == churn_val].sample(min(1000, sum(df['Churn'] == churn_val)), random_state=1)
    axes[0].scatter(sub['TenureMonths'], sub['MonthlyCharges'],
                    c=color, alpha=0.35, s=12, label=label, linewidths=0)

axes[0].set_xlabel('Tenure (months)', color=MUTED, fontsize=10)
axes[0].set_ylabel('Monthly Charges ($)', color=MUTED, fontsize=10)
axes[0].set_title('Scatter: All Customers', color=TEXT, fontsize=11)
axes[0].legend(fontsize=9, framealpha=0)
axes[0].set_facecolor(BG2)

# Hexbin for density
hb = axes[1].hexbin(df['TenureMonths'], df['MonthlyCharges'],
                    C=df['Churn'], gridsize=22, cmap='RdYlBu_r',
                    reduce_C_function=np.mean, linewidths=0.2)
cb = plt.colorbar(hb, ax=axes[1])
cb.set_label('Churn Probability', color=MUTED, fontsize=9)
cb.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=MUTED, fontsize=8)
axes[1].set_xlabel('Tenure (months)', color=MUTED, fontsize=10)
axes[1].set_ylabel('Monthly Charges ($)', color=MUTED, fontsize=10)
axes[1].set_title('Hexbin: Avg Churn Probability', color=TEXT, fontsize=11)
axes[1].set_facecolor(BG2)

for ax in axes:
    ax.tick_params(colors=MUTED)
    ax.spines[:].set_color(BORDER)

plt.tight_layout()
plt.savefig(OUT + 'eda_04_tenure_charges_scatter.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 4: Scatter / hexbin")

# ════════════════════════════════════════════════════════════
# CHART 5 — Tenure cohort churn waterfall
# ════════════════════════════════════════════════════════════
df['TenureBand'] = pd.cut(df['TenureMonths'],
    bins=[0,6,12,24,36,48,60,72],
    labels=['0–6m','7–12m','13–24m','25–36m','37–48m','49–60m','61–72m'])

cohort = df.groupby('TenureBand', observed=True)['Churn'].agg(['mean','count']).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 6), facecolor=BG)
ax2 = ax1.twinx()

bar_colors = [RED if v > 0.25 else YELLOW if v > 0.15 else GREEN for v in cohort['mean']]
bars = ax1.bar(range(len(cohort)), cohort['mean'] * 100, color=bar_colors,
               alpha=0.75, edgecolor=BG, linewidth=0.8, width=0.6, zorder=2)
for b, v in zip(bars, cohort['mean']):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
             f'{v:.1%}', ha='center', va='bottom', fontsize=9,
             color=TEXT, fontweight='bold')

ax2.plot(range(len(cohort)), cohort['count'], color=ACCENT,
         marker='o', markersize=7, linewidth=2, zorder=3)
ax2.fill_between(range(len(cohort)), cohort['count'], alpha=0.1, color=ACCENT)

ax1.set_xticks(range(len(cohort)))
ax1.set_xticklabels(cohort['TenureBand'], fontsize=10)
ax1.set_ylabel('Churn Rate (%)', color=MUTED, fontsize=11)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'{x:.0f}%'))
ax2.set_ylabel('Number of Customers', color=ACCENT, fontsize=11)
ax2.tick_params(axis='y', colors=ACCENT)
ax1.tick_params(colors=MUTED)
ax1.set_facecolor(BG2)
ax1.grid(axis='y', alpha=0.25)
ax1.set_title('Churn Rate by Tenure Cohort  ·  Bar = Churn%   Line = Customer Count',
              color=TEXT, fontsize=12, pad=12)

# Annotation
ax1.annotate('High risk:\nnew customers', xy=(0, cohort['mean'].iloc[0]*100),
             xytext=(1.2, cohort['mean'].iloc[0]*100 + 5),
             fontsize=8, color=RED,
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

plt.tight_layout()
plt.savefig(OUT + 'eda_05_tenure_cohort_churn.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 5: Tenure cohort waterfall")

# ════════════════════════════════════════════════════════════
# CHART 6 — Service adoption vs churn (stacked bars)
# ════════════════════════════════════════════════════════════
services = ['OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies']

rates = {}
for svc in services:
    sub = df[df[svc].isin(['Yes','No'])]
    rates[svc] = {
        'With service':    sub[sub[svc]=='Yes']['Churn'].mean(),
        'Without service': sub[sub[svc]=='No']['Churn'].mean(),
    }

rates_df = pd.DataFrame(rates).T
rates_df.columns.name = None

fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
x    = np.arange(len(services))
w    = 0.35
bars_with    = ax.bar(x - w/2, rates_df['With service']   * 100, w,
                      label='With service',    color=GREEN, alpha=0.8, edgecolor=BG)
bars_without = ax.bar(x + w/2, rates_df['Without service'] * 100, w,
                      label='Without service', color=RED,   alpha=0.8, edgecolor=BG)

for b in list(bars_with) + list(bars_without):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
            f'{b.get_height():.1f}%', ha='center', va='bottom', fontsize=8, color=TEXT)

ax.set_xticks(x)
ax.set_xticklabels([s.replace('Online','Online\n').replace('Device','Device\n')
                    .replace('Tech','Tech\n').replace('Streaming','Streaming\n')
                    for s in services], fontsize=9)
ax.set_ylabel('Churn Rate (%)', color=MUTED, fontsize=11)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'{x:.0f}%'))
ax.set_title('Service Adoption Impact on Churn Rate\n(Protective services reduce churn)',
             color=TEXT, fontsize=13, pad=12)
ax.legend(fontsize=10, framealpha=0)
ax.set_facecolor(BG2); ax.grid(axis='y', alpha=0.25)
ax.tick_params(colors=MUTED)

plt.tight_layout()
plt.savefig(OUT + 'eda_06_service_churn_impact.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 6: Service adoption vs churn")

# ════════════════════════════════════════════════════════════
# CHART 7 — Monthly charges distribution + KDE
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17, 6), facecolor=BG)
fig.suptitle('Financial Metrics — Churn Analysis', fontsize=14,
             color=TEXT, fontweight='bold')

for ax, col, label in zip(
    axes,
    ['MonthlyCharges','TotalCharges','TenureMonths'],
    ['Monthly Charges ($)','Total Charges ($)','Tenure (months)']
):
    for churn_val, color, lbl in [(0, BLUE, 'Retained'), (1, RED, 'Churned')]:
        data = df[df['Churn'] == churn_val][col].dropna()
        ax.hist(data, bins=35, density=True, alpha=0.45,
                color=color, label=lbl, edgecolor='none')
        # KDE line
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data, bw_method=0.25)
        xs  = np.linspace(data.min(), data.max(), 200)
        ax.plot(xs, kde(xs), color=color, linewidth=2)
        ax.axvline(data.median(), color=color, linestyle='--',
                   linewidth=1.2, alpha=0.7)

    ax.set_xlabel(label, color=MUTED, fontsize=10)
    ax.set_ylabel('Density', color=MUTED, fontsize=10)
    ax.legend(fontsize=8, framealpha=0)
    ax.set_facecolor(BG2); ax.grid(alpha=0.2)
    ax.tick_params(colors=MUTED)
    ax.spines[:].set_color(BORDER)

plt.tight_layout()
plt.savefig(OUT + 'eda_07_financial_distributions.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 7: Financial distributions")

# ════════════════════════════════════════════════════════════
# CHART 8 — Churn risk matrix (contract × internet service)
# ════════════════════════════════════════════════════════════
pivot = df.pivot_table(values='Churn', index='Contract',
                       columns='InternetService', aggfunc='mean')

fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
               vmin=0, vmax=0.6, interpolation='nearest')
plt.colorbar(im, ax=ax, label='Churn Rate', fraction=0.04)

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=11, color=TEXT)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=11, color=TEXT)
ax.set_title('Churn Risk Matrix\nContract Type × Internet Service', 
             color=TEXT, fontsize=13, pad=12)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        ax.text(j, i, f'{val:.1%}', ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if val > 0.3 else 'black')

ax.set_facecolor(BG2)
plt.tight_layout()
plt.savefig(OUT + 'eda_08_churn_risk_matrix.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ Chart 8: Risk matrix")

# ── Summary stats ────────────────────────────────────────────
print("\n── EDA Summary ──────────────────────────────────────")
print(f"Total customers:      {len(df):,}")
print(f"Churn rate:           {df['Churn'].mean():.1%}")
print(f"Avg tenure (churned): {df[df['Churn']==1]['TenureMonths'].mean():.1f} months")
print(f"Avg tenure (retained):{df[df['Churn']==0]['TenureMonths'].mean():.1f} months")
print(f"Avg monthly (churned):{df[df['Churn']==1]['MonthlyCharges'].mean():.2f}")
print(f"Missing TotalCharges: {df['TotalCharges'].isna().sum()}")
print(f"\nChurn rate by contract:")
print(df.groupby('Contract')['Churn'].mean().round(3).to_string())
print(f"\nChurn rate by internet service:")
print(df.groupby('InternetService')['Churn'].mean().round(3).to_string())
print("\nAll 8 EDA charts saved to outputs/")
