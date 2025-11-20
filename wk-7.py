# Install (run once in your environment):
# pip install aif360==0.5.0 scikit-learn matplotlib seaborn pandas

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aif360.datasets import CompasDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import DisparateImpactRemover
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# 1) Load COMPAS dataset
compas = CompasDataset(
    features_to_drop=['days_b_screening_arrest', 'c_jail_in', 'c_jail_out']  # optional cleanup
)

# Define privileged/unprivileged groups
privileged_groups = [{'race': 1}]      # AIF360 encodes race==1 as Caucasian
unprivileged_groups = [{'race': 0}]    # race==0 as Non-Caucasian

# 2) Use COMPAS’s provided risk score as the “prediction”
# The dataset includes 'score_text' (Low/Medium/High). Convert to binary "High Risk".
df = compas.convert_to_dataframe()[0].copy()
df['pred'] = (df['score_text'] == 'High').astype(int)
df['label'] = compas.labels.ravel()  # 1 = recidivated within 2 years

# Build a BinaryLabelDataset for metrics
from aif360.datasets import BinaryLabelDataset
bld = BinaryLabelDataset(
    favorable_label=0, unfavorable_label=1,  # unfavorable = recidivate
    df=df,
    label_names=['label'],
    protected_attribute_names=['race'],
    scores=None
)
# Add predictions into a separate dataset
pred_bld = BinaryLabelDataset(
    favorable_label=0, unfavorable_label=1,
    df=df.rename(columns={'pred':'label'}),
    label_names=['label'],
    protected_attribute_names=['race'],
    scores=None
)

# 3) Compute fairness and error metrics
cm = ClassificationMetric(
    bld, pred_bld,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

metrics = {
    'false_positive_rate_unpriv': cm.false_positive_rate(True),
    'false_positive_rate_priv': cm.false_positive_rate(False),
    'false_negative_rate_unpriv': cm.false_negative_rate(True),
    'false_negative_rate_priv': cm.false_negative_rate(False),
    'precision_unpriv': cm.precision(True),
    'precision_priv': cm.precision(False),
    'selection_rate_unpriv': cm.selection_rate(True),
    'selection_rate_priv': cm.selection_rate(False),
    'disparate_impact': BinaryLabelDatasetMetric(
        pred_bld,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    ).disparate_impact()
}

print("Metrics:", pd.Series(metrics))

# 4) Visualizations: FPR/FNR disparity by race
def group_stats(df, group_col='race', y_true='label', y_pred='pred'):
    out = []
    for g in sorted(df[group_col].unique()):
        sub = df[df[group_col] == g]
        tn, fp, fn, tp = confusion_matrix(sub[y_true], sub[y_pred]).ravel()
        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        out.append({'group': g, 'FPR': fpr, 'FNR': fnr, 'Precision': prec, 'N': len(sub)})
    return pd.DataFrame(out)

gstats = group_stats(df)
# Map AIF360 race encoding to labels
race_map = {0: 'Non-Caucasian', 1: 'Caucasian'}
gstats['group'] = gstats['group'].map(race_map)

plt.figure(figsize=(8,5))
sns.barplot(data=gstats.melt(id_vars=['group'], value_vars=['FPR','FNR']),
            x='variable', y='value', hue='group')
plt.title('Error rate disparities by race (COMPAS High Risk)')
plt.ylabel('Rate')
plt.xlabel('Metric')
plt.legend(title='Group')
plt.tight_layout()
plt.show()

# 5) Remediation example: Disparate Impact Remover (pre-processing)
dir_remover = DisparateImpactRemover(repair_level=1.0, sensitive_attribute='race')
repaired = dir_remover.fit_transform(compas)

# Train a simple model on repaired features to compare
# Create train labels and features (original vs repaired)
X_orig = compas.features
y = compas.labels.ravel()
X_rep = repaired.features

clf_orig = LogisticRegression(max_iter=200).fit(X_orig, y)
clf_rep = LogisticRegression(max_iter=200).fit(X_rep, y)

# Predictions (convert to BinaryLabelDataset for metrics)
def to_bld_like(orig_bld, y_pred):
    dfp = orig_bld.convert_to_dataframe()[0].copy()
    dfp['label'] = y_pred
    return BinaryLabelDataset(
        favorable_label=0, unfavorable_label=1,
        df=dfp, label_names=['label'],
        protected_attribute_names=['race']
    )

pred_orig = to_bld_like(bld, (clf_orig.predict(X_orig) > 0.5).astype(int))
pred_rep = to_bld_like(bld, (clf_rep.predict(X_rep) > 0.5).astype(int))

cm_orig = ClassificationMetric(bld, pred_orig, unprivileged_groups, privileged_groups)
cm_rep = ClassificationMetric(bld, pred_rep, unprivileged_groups, privileged_groups)

print("Original FPR gap:", cm_orig.false_positive_rate(True) - cm_orig.false_positive_rate(False))
print("Repaired FPR gap:", cm_rep.false_positive_rate(True) - cm_rep.false_positive_rate(False))

# Plot disparate impact before/after
def disparate_impact(dset):
    return BinaryLabelDatasetMetric(dset, unprivileged_groups, privileged_groups).disparate_impact()

di_orig = disparate_impact(pred_orig)
di_rep = disparate_impact(pred_rep)

plt.figure(figsize=(6,4))
sns.barplot(x=['Original','Repaired'], y=[di_orig, di_rep], palette='viridis')
plt.title('Disparate Impact before vs after repair')
plt.ylabel('DI (>=0.8 often considered acceptable)')
plt.tight_layout()
plt.show()
