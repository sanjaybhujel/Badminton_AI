# =============================================================================
#  Explainable Machine Learning for Badminton Match Prediction
#  Sanjay Bhujel | B01802556 | MSc IT with Data Analytics | UWS
#  Supervisor: Muhammad Yasir Adnan
#
#  DATASET : BWF World Tour (sanderp/badminton-bwf-world-tour on Kaggle)
#            Local file: ms.csv  (3,761 rows x 38 columns)
#
#  INSTALL :
#    pip install pandas numpy scikit-learn xgboost shap lime statsmodels
#               matplotlib seaborn joblib kagglehub
#
#  RUN     :
#    python badminton_final_pipeline.py
#
#  All outputs saved to  ./outputs/
# =============================================================================
import argparse
import os
import warnings
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and explain badminton match prediction models."
    )
    parser.add_argument(
        "--data",
        default="ms.csv",
        help="Path to the men's singles CSV dataset. Default: ms.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where charts, reports, and models are saved. Default: outputs",
    )
    parser.add_argument(
        "--skip-explainability",
        action="store_true",
        help="Skip SHAP and LIME sections for faster runs or minimal installs.",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip randomized hyperparameter tuning for faster smoke tests.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for cross-validation and tuning. Default: 1",
    )
    return parser.parse_args()


ARGS = parse_args()
DATA_PATH = Path(ARGS.data)
OUTPUT_DIR = Path(ARGS.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def output_path(filename):
    return OUTPUT_DIR / filename


def positive_class_shap_values(values):
    if isinstance(values, list):
        return values[1] if len(values) > 1 else values[0]
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def positive_class_expected_value(value):
    if isinstance(value, list):
        return value[1] if len(value) > 1 else value[0]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.ravel()
    return flat[1] if flat.size > 1 else flat[0]

# ── Plot styling ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#F8F8F8',
    'axes.edgecolor': '#CCCCCC', 'axes.linewidth': 0.8,
    'grid.color': '#E0E0E0', 'grid.linewidth': 0.6,
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 11, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.dpi': 150,
})
BLUE, ORANGE, GREEN = '#2E5090', '#E8593C', '#2CA02C'
PALETTE = [BLUE, ORANGE, GREEN, '#FF7F0E', '#9467BD']

print("=" * 70)
print("  Explainable ML — Badminton Match Prediction & Performance Analytics")
print("  Sanjay Bhujel | B01802556 | MSc IT with Data Analytics | UWS")
print("=" * 70)

# =============================================================================
# CELL 1 — Load Dataset
# =============================================================================
print("\n[1/10] Loading BWF World Tour dataset …")

# ── Option B: Download directly from Kaggle ───────────────────────────────────
# try:
#     import kagglehub
#     kpath = kagglehub.dataset_download("sanderp/badminton-bwf-world-tour")
#     import glob
#     csv_files = glob.glob(os.path.join(kpath, "*.csv"))
#     DATA_PATH = csv_files[0]
#     print(f"  Downloaded to: {DATA_PATH}")
# except Exception as e:
#     print(f"  Kaggle download failed ({e}), falling back to local ms.csv")
#     DATA_PATH = 'ms.csv'

try:
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  Loaded: {DATA_PATH}")
    print(f"  Raw shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
except FileNotFoundError:
    raise FileNotFoundError(
        f"\n  ERROR: '{DATA_PATH}' not found.\n"
        "  Place ms.csv in the same folder as this script,\n"
        "  pass --data /path/to/ms.csv, or uncomment the Kaggle download block above.\n"
    )

# =============================================================================
# CELL 2 — Dataset Inspection & Pre-Cleaning EDA
# =============================================================================
print("\n[2/10] Dataset inspection & EDA …")

print(f"\n  Columns ({df_raw.shape[1]}):")
for i, col in enumerate(df_raw.columns):
    dtype = str(df_raw[col].dtype)
    n_null = df_raw[col].isnull().sum()
    print(f"    [{i:02d}] {col:<50} {dtype:<10} nulls={n_null}")

# ── Winner column analysis ────────────────────────────────────────────────────
print(f"\n  'winner' value counts:")
print(f"    {df_raw['winner'].value_counts().to_dict()}")
print("    → 0 = retirement/walkover (will be removed)")
print("    → 1 = team one wins")
print("    → 2 = team two wins")

# ── Missing values chart (pre-clean) ─────────────────────────────────────────
missing     = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(1)
missing_df  = pd.DataFrame({'count': missing, 'pct': missing_pct})
missing_df  = missing_df[missing_df['count'] > 0].sort_values('pct', ascending=False)

if not missing_df.empty:
    fig, ax = plt.subplots(figsize=(10, max(3, len(missing_df) * 0.5)))
    bars = ax.barh(missing_df.index, missing_df['pct'],
                   color=ORANGE, edgecolor='white', height=0.6)
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=9)
    ax.set_xlabel('Missing (%)')
    ax.set_title('Missing Values by Feature (raw dataset)')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path('eda_missing_values.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Missing value summary:\n{missing_df.to_string()}")
    print("  Saved: outputs/eda_missing_values.png")
    print("\n  NOTE: game_3_score/stats missing ~65% — expected (2-set matches).")
    print("        game_2_score/stats missing ~1%  — single-game matches (nb_sets=1).")
else:
    print("  No missing values found.")

# =============================================================================
# CELL 3 — Data Cleaning
# =============================================================================
print("\n[3/10] Data cleaning …")

df = df_raw.copy()

# ── Remove retirements / walkovers (winner == 0) ──────────────────────────────
retired_count = (df['winner'] == 0).sum()
df = df[df['winner'].isin([1, 2])].copy()
print(f"  Removed {retired_count} retirement/walkover matches (winner=0)")
print(f"  Remaining: {df.shape[0]:,} complete matches")

# ── Remove duplicates ─────────────────────────────────────────────────────────
before = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"  Duplicates removed: {before - df.shape[0]}")

# ── Binary target variable ────────────────────────────────────────────────────
df['team_one_wins'] = (df['winner'] == 1).astype(int)
print(f"\n  Target variable 'team_one_wins' distribution:")
print(f"    {df['team_one_wins'].value_counts().to_dict()}")
balance = df['team_one_wins'].value_counts(normalize=True).round(3)
print(f"    Balance: Win={balance[1]:.1%}  Loss={balance[0]:.1%} — well balanced, no resampling needed")

# ── Outlier capping (IQR) on numeric columns ──────────────────────────────────
print("\n  Outlier capping (IQR method):")
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c not in ['winner', 'team_one_wins']]
capped_total = 0
for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    if n_out > 0:
        df[col] = df[col].clip(lower=lower, upper=upper)
        capped_total += n_out
        print(f"    {col}: {n_out} outlier(s) capped to [{lower:.1f}, {upper:.1f}]")
if capped_total == 0:
    print("    No outliers found.")

# =============================================================================
# CELL 4 — Feature Engineering
# =============================================================================
print("\n[4/10] Feature engineering …")

def parse_score(s):
    """Parse '21-18' → (21, 18); returns (NaN, NaN) on failure."""
    try:
        a, b = str(s).split('-')
        return int(a), int(b)
    except Exception:
        return np.nan, np.nan

df[['g1_t1', 'g1_t2']] = df['game_1_score'].apply(
    lambda x: pd.Series(parse_score(x)))

# ── Group 1: Performance differential features ────────────────────────────────
df['total_points_diff']  = df['team_one_total_points']  - df['team_two_total_points']
df['game_points_diff']   = df['team_one_game_points']   - df['team_two_game_points']
df['consec_points_diff'] = (df['team_one_most_consecutive_points']
                            - df['team_two_most_consecutive_points'])
df['g1_score_diff']      = df['g1_t1'] - df['g1_t2']
df['consec_g1_diff']     = (df['team_one_most_consecutive_points_game_1']
                            - df['team_two_most_consecutive_points_game_1'])
df['game_pts_g1_diff']   = (df['team_one_game_points_game_1']
                            - df['team_two_game_points_game_1'])
print("  + total_points_diff, game_points_diff, consec_points_diff")
print("  + g1_score_diff, consec_g1_diff, game_pts_g1_diff")

# ── Group 2: Absolute performance features ────────────────────────────────────
print("  + team_one_total_points, team_two_total_points (retained)")
print("  + team_one_game_points,  team_two_game_points  (retained)")
print("  + team_one_most_consecutive_points, team_two_most_consecutive_points (retained)")

# ── Group 3: Game 1 context features ─────────────────────────────────────────
df['g1_total']      = df['g1_t1'] + df['g1_t2']
df['t1_win_pct_g1'] = df['g1_t1'] / df['g1_total'].replace(0, np.nan)
print("  + g1_total (rally intensity), t1_win_pct_g1 (game 1 points share)")

# ── Group 4: Contextual ordinal features ─────────────────────────────────────
round_order = {
    'Qualification round of 32':   0,
    'Qualification round of 16':   1,
    'Qualification quarter final': 2,
    'Round of 64':                 3,
    'Round of 32':                 4,
    'Round 1':                     4,
    'Round 2':                     5,
    'Round 3':                     6,
    'Round of 16':                 5,
    'Quarter final':               6,
    'Semi final':                  7,
    'Final':                       8,
}
tier_order = {
    'BWF Tour Super 100':             1,
    'HSBC BWF World Tour Super 300':  2,
    'HSBC BWF World Tour Super 500':  3,
    'HSBC BWF World Tour Super 750':  4,
    'HSBC BWF World Tour Super 1000': 5,
    'HSBC BWF World Tour Finals':     6,
}
df['round_num']       = df['round'].map(round_order)
df['tournament_tier'] = df['tournament_type'].map(tier_order)
print("  + round_num (ordinal 0–8), tournament_tier (ordinal 1–6)")

# ── Final feature list ────────────────────────────────────────────────────────
FEATURES = [
    # Differential features
    'total_points_diff',
    'game_points_diff',
    'consec_points_diff',
    'g1_score_diff',
    'consec_g1_diff',
    'game_pts_g1_diff',
    # Absolute performance
    'team_one_total_points',
    'team_two_total_points',
    'team_one_game_points',
    'team_two_game_points',
    'team_one_most_consecutive_points',
    'team_two_most_consecutive_points',
    # Game 1 context
    'g1_total',
    't1_win_pct_g1',
    # Match context
    'round_num',
    'tournament_tier',
    'nb_sets',
]
TARGET = 'team_one_wins'

print(f"\n  Total features: {len(FEATURES)}")
print(f"  Feature list: {FEATURES}")


def predict_match(row, model):
    """Predict win probability for a single match row (DataFrame row)."""
    row_arr = row.values.reshape(1, -1)
    prob = model.predict_proba(row_arr)[0, 1]
    print(f"Team 1 win probability: {prob:.3f}")
    return prob


# =============================================================================
# CELL 5 — Preprocessing: Drop NAs, Split, Scale
# =============================================================================
print("\n[5/10] Preprocessing …")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler

# FIX: dropna BEFORE VIF check so statsmodels doesn't see NaN rows
model_df = df[FEATURES + [TARGET]].dropna().reset_index(drop=True)
print(f"  Rows after dropna: {model_df.shape[0]:,}  (dropped {df.shape[0] - model_df.shape[0]} NaN rows)")

X = model_df[FEATURES]

# ── VIF check (on clean, NaN-free data) ──────────────────────────────────────
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"]     = [
    variance_inflation_factor(X.values.astype(float), i)
    for i in range(X.shape[1])
]
print("\n═══ VIF TABLE ═══")
print(vif_data.sort_values("VIF", ascending=False).to_string(index=False))

y = model_df[TARGET]

print(f"\n  Feature matrix : {X.shape}")
print(f"  Target balance :\n{y.value_counts(normalize=True).round(3).to_string()}")

# ── Stratified 80/20 split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# StandardScaler for Logistic Regression
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

joblib.dump(scaler, output_path('scaler.pkl'))

print(f"\n  Train: {X_train.shape[0]:,} samples")
print(f"  Test : {X_test.shape[0]:,} samples")
print(f"  Test  target balance: {y_test.value_counts().to_dict()}")
print("  Saved: outputs/scaler.pkl")

with open(output_path('selected_features.txt'), 'w') as f:
    f.write('\n'.join(FEATURES))
print("  Saved: outputs/selected_features.txt")

# ── EDA: Class distribution ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

counts = y.value_counts()
bars = axes[0].bar(['Team 2 wins (0)', 'Team 1 wins (1)'],
                   [counts[0], counts[1]],
                   color=[ORANGE, BLUE], edgecolor='white', width=0.5)
axes[0].bar_label(bars, padding=4)
axes[0].set_title('Match Outcome Distribution')
axes[0].set_ylabel('Count')

print("\n  NOTE: Strong correlations expected among point-based features (important for SHAP interpretation).")
top12 = pd.concat([X, y], axis=1).corr()[TARGET].abs().nlargest(13).index
sns.heatmap(pd.concat([X, y], axis=1)[top12].corr(), ax=axes[1],
            cmap='coolwarm', annot=True, fmt='.2f',
            linewidths=0.4, annot_kws={'size': 7}, square=True)
axes[1].set_title('Correlation Heatmap — Top 12 Features')
axes[1].tick_params(axis='x', rotation=45, labelsize=8)
axes[1].tick_params(axis='y', labelsize=8)

plt.tight_layout()
plt.savefig(output_path('eda_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/eda_overview.png")

# ── Feature distributions ─────────────────────────────────────────────────────
diff_features = ['total_points_diff', 'game_points_diff', 'consec_points_diff',
                 'g1_score_diff', 'consec_g1_diff', 'game_pts_g1_diff']
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, feat in zip(axes.flatten(), diff_features):
    for label, color in [(0, ORANGE), (1, BLUE)]:
        subset = model_df[model_df[TARGET] == label][feat].dropna()
        ax.hist(subset, bins=25, alpha=0.6, color=color, edgecolor='white',
                label='Team 2 wins' if label == 0 else 'Team 1 wins')
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linewidth=0.4)
plt.suptitle('Differential Feature Distributions by Match Outcome',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path('eda_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/eda_feature_distributions.png")

# ── Boxplot for strongest feature ─────────────────────────────────────────────
plt.figure(figsize=(6, 4))
sns.boxplot(x=TARGET, y='total_points_diff', data=model_df,
            palette=[ORANGE, BLUE])
plt.xticks([0, 1], ['Team 2 wins', 'Team 1 wins'])
plt.title('Total Points Difference by Match Outcome')
plt.xlabel('Match Outcome')
plt.ylabel('Total Points Difference')
plt.tight_layout()
plt.savefig(output_path('eda_boxplot_total_points.png'), dpi=150)
plt.close()
print("  Saved: outputs/eda_boxplot_total_points.png")

# =============================================================================
# CELL 6 — Model Training
# FIX: Define ALL individual models + Stacking so that:
#   - fitted['Random Forest'] exists for feature importance (was KeyError)
#   - Each model uses the correct (scaled / unscaled) feature matrix
#   - Confusion matrix and learning curve subplots are sized dynamically
# =============================================================================
print("\n[6/10] Training models …")

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    booster = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42)
    booster_name = 'XGBoost'
    print("  Using XGBoost for gradient boosting.")
except Exception as e:
    booster = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42)
    booster_name = 'Gradient Boosting'
    print(f"  XGBoost unavailable ({e.__class__.__name__}) — using sklearn GradientBoostingClassifier.")

from sklearn.ensemble import StackingClassifier

# FIX: individual models AND stacking all live in one dict.
# 'uses_scaler' flag tells the eval loop which X to pass.
lr_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

stacking_base = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
]
if booster_name == 'XGBoost':
    from xgboost import XGBClassifier as _XGB
    stacking_base.append(('xgb', _XGB(eval_metric='logloss', random_state=42)))
else:
    stacking_base.append(('gb', GradientBoostingClassifier(random_state=42)))

models = {
    'Logistic Regression': (lr_model,      True),   # (estimator, uses_scaler)
    'Random Forest':       (rf_model,      False),
    booster_name:          (booster,       False),
    'Stacking Ensemble':   (
        StackingClassifier(
            estimators=stacking_base,
            final_estimator=LogisticRegression(),
            cv=5,
        ),
        False,
    ),
}

# =============================================================================
# CELL 7 — Model Evaluation
# FIX: ECE function defined OUTSIDE loop; ECE scores printed AFTER loop;
#      calibration moved AFTER loop with cv='prefit'.
# =============================================================================
print("\n[7/10] Evaluating models …")

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              RocCurveDisplay, confusion_matrix)


# FIX: define ECE outside the loop so it is always available
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bins   = np.linspace(0, 1, n_bins + 1)
    binids = np.minimum(np.digitize(y_prob, bins) - 1, n_bins - 1)
    ece    = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            acc  = np.mean(np.array(y_true)[mask])
            conf = np.mean(np.array(y_prob)[mask])
            ece += abs(acc - conf) * np.sum(mask) / len(y_true)
    return ece


results = {}
fitted  = {}
preds   = {}

for name, (model, uses_scaler) in models.items():
    X_tr = X_train_sc if uses_scaler else X_train
    X_te = X_test_sc  if uses_scaler else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    results[name] = {
        'Accuracy':  round(accuracy_score(y_test, y_pred),                   4),
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        'F1-Score':  round(f1_score(y_test, y_pred,        zero_division=0), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob),                    4),
    }
    fitted[name] = model
    preds[name]  = (y_pred, y_prob)

    print(f"\n  ── {name} ──")
    print(classification_report(y_test, y_pred,
                                 target_names=['Team 2 wins (0)', 'Team 1 wins (1)'],
                                 zero_division=0))

# FIX: ECE printed AFTER loop — preds is now fully populated
print("\n═══ ECE SCORES ═══")
for name, (_, y_prob) in preds.items():
    ece = expected_calibration_error(np.array(y_test), np.array(y_prob))
    print(f"  {name:<30}: ECE = {ece:.4f}")

results_df = pd.DataFrame(results).T

# ── Calibration curves ────────────────────────────────────────────────────────
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


def make_prefit_calibrator(model):
    """Return a calibration wrapper for an already fitted model.

    scikit-learn 1.6 replaced cv='prefit' with FrozenEstimator. The fallback
    keeps the script usable on older versions.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
    except Exception:
        return CalibratedClassifierCV(model, method='isotonic', cv='prefit')


calibrated_models = {}
for name, (model, uses_scaler) in models.items():
    calibrated = make_prefit_calibrator(fitted[name])
    X_cal = X_test_sc if uses_scaler else X_test
    calibrated.fit(X_cal, y_test)          # fits the isotonic layer on held-out data
    calibrated_models[name] = (calibrated, uses_scaler)

plt.figure(figsize=(6, 5))
for i, (name, (cal_model, uses_scaler)) in enumerate(calibrated_models.items()):
    X_te  = X_test_sc if uses_scaler else X_test
    prob  = cal_model.predict_proba(X_te)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
    plt.plot(mean_pred, frac_pos, marker='o', color=PALETTE[i], label=name)

plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.xlabel("Predicted Probability")
plt.ylabel("Actual Probability")
plt.title("Calibration Curves — All Models")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(output_path("calibration_curve.png"), dpi=150)
plt.close()
print("  Saved: outputs/calibration_curve.png")

print("\n  ═══ MODEL COMPARISON SUMMARY ═══")
print(results_df.to_string())
results_df.to_csv(output_path('model_comparison.csv'))
print("  Saved: outputs/model_comparison.csv")

# ── Bar chart ─────────────────────────────────────────────────────────────────
n_models = len(results_df)
fig, ax  = plt.subplots(figsize=(11, 5))
x        = np.arange(len(results_df.columns))
width    = 0.8 / n_models                         # FIX: dynamic bar width
for i, (name, row) in enumerate(results_df.iterrows()):
    bars = ax.bar(x + i * width, row.values, width,
                  label=name, color=PALETTE[i % len(PALETTE)],
                  edgecolor='white', alpha=0.9)
    ax.bar_label(bars, fmt='%.4f', padding=2, fontsize=7, rotation=90)
ax.set_xticks(x + width * (n_models - 1) / 2)
ax.set_xticklabels(results_df.columns)
ax.set_ylim(0.90, 1.02)
ax.set_title('Model Performance Comparison — BWF Badminton Dataset')
ax.set_ylabel('Score')
ax.legend()
ax.grid(axis='y', linewidth=0.4)
plt.tight_layout()
plt.savefig(output_path('model_comparison_barplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/model_comparison_barplot.png")

# ── ROC curves ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
for i, (name, (_, y_prob)) in enumerate(preds.items()):
    RocCurveDisplay.from_predictions(
        y_test, y_prob,
        name=f"{name} (AUC={results[name]['ROC-AUC']})",
        ax=ax, color=PALETTE[i % len(PALETTE)])
ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Random classifier')
ax.set_title('ROC Curves — All Models')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(output_path('roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/roc_curves.png")

# ── Confusion matrices — FIX: dynamic subplot grid ───────────────────────────
ncols = min(n_models, 4)
nrows = (n_models + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
axes_flat = np.array(axes).flatten()

for i, (name, (y_pred, _)) in enumerate(preds.items()):
    ax = axes_flat[i]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['T2 wins', 'T1 wins'],
                yticklabels=['T2 wins', 'T1 wins'],
                cbar=False, linewidths=0.5, annot_kws={'size': 12})
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

# Hide any unused axes
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle('Confusion Matrices — BWF Badminton Dataset', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path('confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/confusion_matrices.png")

# ── 5-fold Cross-validation scores ───────────────────────────────────────────
print("\n  ═══ 5-FOLD CROSS-VALIDATED ROC-AUC ═══")
cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, (model, uses_scaler) in models.items():
    X_cv   = X_train_sc if uses_scaler else X_train
    scores = cross_val_score(fitted[name], X_cv, y_train,
                              cv=cv, scoring='roc_auc', n_jobs=ARGS.n_jobs)
    cv_results[name] = {'CV Mean AUC': round(scores.mean(), 4),
                         'CV Std':      round(scores.std(),  4)}
    print(f"  {name:30s}: {scores.mean():.4f} ± {scores.std():.4f}")

cv_df = pd.DataFrame(cv_results).T
cv_df.to_csv(output_path('cv_scores.csv'))
print("  Saved: outputs/cv_scores.csv")

# ── Learning curves — FIX: dynamic subplot grid ───────────────────────────────
from sklearn.model_selection import learning_curve

ncols_lc = min(n_models, 4)
nrows_lc = (n_models + ncols_lc - 1) // ncols_lc
fig, axes = plt.subplots(nrows_lc, ncols_lc,
                          figsize=(ncols_lc * 4.5, nrows_lc * 4))
axes_flat = np.array(axes).flatten()

for i, (name, (model, uses_scaler)) in enumerate(models.items()):
    X_lc = X_train_sc if uses_scaler else X_train
    sizes, tr_sc, val_sc = learning_curve(
        fitted[name], X_lc, y_train, cv=cv, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=ARGS.n_jobs)
    ax = axes_flat[i]
    ax.plot(sizes, tr_sc.mean(1),  'o-', color=BLUE,   label='Train AUC')
    ax.plot(sizes, val_sc.mean(1), 's-', color=ORANGE,  label='Val AUC')
    ax.fill_between(sizes,
                    tr_sc.mean(1)  - tr_sc.std(1),
                    tr_sc.mean(1)  + tr_sc.std(1),
                    alpha=0.12, color=BLUE)
    ax.fill_between(sizes,
                    val_sc.mean(1) - val_sc.std(1),
                    val_sc.mean(1) + val_sc.std(1),
                    alpha=0.12, color=ORANGE)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Training samples')
    ax.set_ylabel('ROC-AUC')
    ax.set_ylim(0.88, 1.01)
    ax.legend(fontsize=9)
    ax.grid(linewidth=0.4)

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.suptitle('Learning Curves — 5-fold CV', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(output_path('learning_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/learning_curves.png")

# ── Random Forest feature importances ─────────────────────────────────────────
# FIX: fitted['Random Forest'] now guaranteed to exist
rf_fitted   = fitted['Random Forest']
importances = pd.Series(rf_fitted.feature_importances_,
                          index=FEATURES).nlargest(17).sort_values()
fig, ax = plt.subplots(figsize=(9, 7))
colors  = [BLUE if v > 0.05 else '#7aabd4' for v in importances.values]
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Random Forest — Feature Importances (Gini)')
ax.set_xlabel('Importance')
ax.axvline(x=0.05, color=ORANGE, linestyle='--', linewidth=0.8, alpha=0.7,
            label='0.05 threshold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(output_path('feature_importance_rf.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: outputs/feature_importance_rf.png")

# =============================================================================
# CELL 8 — SHAP Explainability
# FIX: restructured if/else into proper if … else block;
#      interaction values only computed for tree models;
#      win_idx indentation corrected.
# =============================================================================
print("\n[8/10] SHAP explainability …")

if ARGS.skip_explainability:
    print("  Skipped by --skip-explainability.")
else:
    try:
        import shap

        tree_explainers = {'Random Forest', 'XGBoost', 'Gradient Boosting'}
        best_name = results_df['ROC-AUC'].idxmax()
        if best_name not in tree_explainers and best_name != 'Logistic Regression':
            explainable_results = {
                k: v for k, v in results.items()
                if k in tree_explainers or k == 'Logistic Regression'
            }
            best_name = max(explainable_results, key=lambda k: explainable_results[k]['ROC-AUC'])
            print(f"  Best overall model is not directly SHAP-compatible; explaining {best_name}.")

        best_model = fitted[best_name]
        best_uses_scaler = models[best_name][1]
        print(f"  SHAP model: {best_name}  (AUC = {results[best_name]['ROC-AUC']})")
    
        X_shap    = X_test_sc if best_uses_scaler else X_test
        X_shap_df = pd.DataFrame(X_shap, columns=FEATURES)
    
        # FIX: clean if/else — no orphaned second if, no misplaced else
        if best_uses_scaler:   # Logistic Regression path
            explainer        = shap.LinearExplainer(best_model, X_train_sc)
            shap_values      = positive_class_shap_values(explainer.shap_values(X_shap))
            interaction_values = None          # not available for linear models

        else:                  # Tree-based path (RF, XGBoost, Stacking)
            explainer   = shap.TreeExplainer(best_model)
            shap_values = positive_class_shap_values(explainer.shap_values(X_shap))
    
            # SHAP interaction values (tree models only)
            try:
                interaction_values = explainer.shap_interaction_values(X_shap)
                plt.figure(figsize=(8, 6))
                shap.summary_plot(interaction_values, X_shap_df,
                                  feature_names=FEATURES, show=False)
                plt.title("SHAP Interaction Effects")
                plt.tight_layout()
                plt.savefig(output_path("shap_interactions.png"), dpi=150)
                plt.close()
                print("  Saved: outputs/shap_interactions.png")
            except Exception as e:
                interaction_values = None
                print(f"  SHAP interaction values skipped: {e}")
    
        ev = positive_class_expected_value(explainer.expected_value)
    
        # ── Global: bar summary ───────────────────────────────────────────────────
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_shap_df, feature_names=FEATURES,
                          plot_type='bar', show=False, max_display=17, color=BLUE)
        plt.title(f'SHAP Feature Importance (bar) — {best_name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path('shap_summary_bar.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: outputs/shap_summary_bar.png")
    
        # ── Global: beeswarm ─────────────────────────────────────────────────────
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_shap_df, feature_names=FEATURES,
                          show=False, max_display=17)
        plt.title(f'SHAP Beeswarm Plot — {best_name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path('shap_summary_beeswarm.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: outputs/shap_summary_beeswarm.png")
    
        # ── Clean beeswarm (top 10, for dissertation) ─────────────────────────────
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_shap_df, feature_names=FEATURES,
                          show=False, max_display=10)
        plt.tight_layout()
        plt.savefig(output_path('shap_beeswarm_clean.png'), dpi=150)
        plt.close()
        print("  Saved: outputs/shap_beeswarm_clean.png")
    
        # ── Dependence plot for top feature ──────────────────────────────────────
        top_feat  = FEATURES[np.abs(shap_values).mean(axis=0).argmax()]
        safe_name = top_feat.replace('/', '_').replace(' ', '_')
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(top_feat, shap_values, X_shap_df,
                              feature_names=FEATURES, show=False)
        plt.title(f'SHAP Dependence — {top_feat}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path(f'shap_dependence_{safe_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: outputs/shap_dependence_{safe_name}.png")
        print(f"  Top SHAP feature: {top_feat}")
    
        # ── Local: force plot — Team 1 win ────────────────────────────────────────
        # FIX: both lines share the same indentation level
        win_indices = np.where(np.array(y_test) == 1)[0]
        win_idx     = win_indices[0] if len(win_indices) > 0 else 0
    
        plt.figure(figsize=(14, 3))
        shap.force_plot(ev, shap_values[win_idx], X_shap_df.iloc[win_idx],
                        feature_names=FEATURES, matplotlib=True, show=False)
        plt.title(f'SHAP Force Plot — Team 1 Win (Match {win_idx})', fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path('shap_local_win_case.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: outputs/shap_local_win_case.png")
    
        # ── Local: force plot — Team 2 win ────────────────────────────────────────
        loss_idx = np.where(np.array(y_test) == 0)[0][0]
        plt.figure(figsize=(14, 3))
        shap.force_plot(ev, shap_values[loss_idx], X_shap_df.iloc[loss_idx],
                        feature_names=FEATURES, matplotlib=True, show=False)
        plt.title(f'SHAP Force Plot — Team 2 Win (Match {loss_idx})', fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path('shap_local_loss_case.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: outputs/shap_local_loss_case.png")
    
        # ── Local: waterfall plot ─────────────────────────────────────────────────
        try:
            shap_exp = shap.Explanation(
                values=shap_values[win_idx], base_values=ev,
                data=X_shap_df.iloc[win_idx].values, feature_names=FEATURES)
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_exp, show=False)
            plt.title(f'SHAP Waterfall — Team 1 Win (Match {win_idx})',
                      fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path('shap_waterfall_win.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("  Saved: outputs/shap_waterfall_win.png")
    
            # ── Borderline case (probability near 0.5) ────────────────────────────
            probs           = best_model.predict_proba(X_shap)[:, 1]
            borderline_idx  = np.argmin(np.abs(probs - 0.5))
            shap_exp_border = shap.Explanation(
                values=shap_values[borderline_idx],
                base_values=ev,
                data=X_shap_df.iloc[borderline_idx].values,
                feature_names=FEATURES)
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_exp_border, show=False)
            plt.title(f'SHAP Waterfall — Borderline Case (Match {borderline_idx})',
                      fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(output_path('shap_waterfall_borderline.png'), dpi=150)
            plt.close()
            print("  Saved: outputs/shap_waterfall_borderline.png")
    
        except Exception as e:
            print(f"  Waterfall skipped: {e}")
    
        # ── SHAP top 10 ranking ───────────────────────────────────────────────────
        shap_imp = pd.Series(np.abs(shap_values).mean(axis=0),
                              index=FEATURES).nlargest(10)
        print("\n  ═══ SHAP TOP 10 FEATURE IMPORTANCE ═══")
        for rank, (feat, val) in enumerate(shap_imp.items(), 1):
            print(f"    {rank:2d}. {feat:<45} {val:.4f}")
    
    except ImportError:
        print("  SHAP not installed — skipping. Install with:  pip install shap")
    except Exception as e:
        print(f"  SHAP skipped because it failed safely: {e}")
    
    # =============================================================================
    # CELL 9 — LIME Local Explanations
    # =============================================================================
print("\n[9/10] LIME local explanations …")

if ARGS.skip_explainability:
    print("  Skipped by --skip-explainability.")
else:
    try:
        import lime, lime.lime_tabular
    
        best_name        = results_df['ROC-AUC'].idxmax()
        best_model       = fitted[best_name]
        best_uses_scaler = models[best_name][1]
        X_lime_tr = X_train_sc if best_uses_scaler else np.array(X_train)
        X_lime_te = X_test_sc  if best_uses_scaler else np.array(X_test)
    
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data = X_lime_tr,
            feature_names = FEATURES,
            class_names   = ['Team 2 wins (0)', 'Team 1 wins (1)'],
            mode          = 'classification',
            random_state  = 42,
        )
    
        # LIME for a Team 1 win
        win_idx  = np.where(np.array(y_test) == 1)[0][0]
        lime_win = lime_explainer.explain_instance(
            data_row    = X_lime_te[win_idx],
            predict_fn  = best_model.predict_proba,
            num_features= 15,
        )
        lime_win.save_to_file(output_path('lime_win_case.html'))
        lime_win.as_pyplot_figure()
        plt.title(f'LIME — Team 1 Win Case | {best_name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path('lime_win_case.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: outputs/lime_win_case.png + .html")
    
        # LIME for a Team 2 win
        loss_idx  = np.where(np.array(y_test) == 0)[0][0]
        lime_loss = lime_explainer.explain_instance(
            data_row    = X_lime_te[loss_idx],
            predict_fn  = best_model.predict_proba,
            num_features= 15,
        )
        lime_loss.save_to_file(output_path('lime_loss_case.html'))
        lime_loss.as_pyplot_figure()
        plt.title(f'LIME — Team 2 Win Case | {best_name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path('lime_loss_case.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: outputs/lime_loss_case.png + .html")
    
    except ImportError:
        print("  LIME not installed — skipping. Install with:  pip install lime")
    except Exception as e:
        print(f"  LIME skipped because it failed safely: {e}")

# =============================================================================
# CELL 10 — Hyperparameter Tuning (best model)
# FIX: tuning explicitly targets the non-stacking best model; scaler applied
#      when tuning Logistic Regression.
# =============================================================================
print("\n[10/10] Hyperparameter tuning …")

from sklearn.model_selection import RandomizedSearchCV

best_name = results_df['ROC-AUC'].idxmax()
print(f"  Best overall model: {best_name}")

if ARGS.skip_tuning:
    final_model = fitted[best_name]
    final_name = best_name
    print("  Skipped by --skip-tuning.")
else:
    # FIX: choose a tunable base model — skip stacking (too slow to tune end-to-end)
    if best_name == 'Stacking Ensemble':
        # Fall back to best individual model for tuning
        individual_results = {k: v for k, v in results.items()
                              if k != 'Stacking Ensemble'}
        tune_name  = max(individual_results, key=lambda k: individual_results[k]['ROC-AUC'])
        print(f"  Stacking not tuned end-to-end — tuning best individual: {tune_name}")
    else:
        tune_name = best_name
    
    tune_uses_scaler = models[tune_name][1]
    print(f"  Tuning: {tune_name}")
    
    if tune_name == 'Logistic Regression':
        param_dist = {
            'C':         [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty':   ['l2'],
            'solver':    ['lbfgs', 'saga'],
            'max_iter':  [500, 1000, 2000],
        }
        base = LogisticRegression(random_state=42)
    
    elif tune_name == booster_name and booster_name == 'XGBoost':
        from xgboost import XGBClassifier as TuneModel
        param_dist = {
            'n_estimators':     [100, 200, 300, 400],
            'max_depth':        [3, 4, 5, 6, 8],
            'learning_rate':    [0.01, 0.05, 0.1, 0.15],
            'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha':        [0, 0.01, 0.1],
            'reg_lambda':       [1, 1.5, 2],
        }
        base = TuneModel(eval_metric='logloss', random_state=42)
    
    elif tune_name == booster_name:   # sklearn GradientBoosting
        param_dist = {
            'n_estimators':     [100, 200, 300],
            'max_depth':        [3, 4, 5, 6],
            'learning_rate':    [0.01, 0.05, 0.1],
            'subsample':        [0.7, 0.8, 0.9, 1.0],
            'min_samples_leaf': [1, 2, 5],
        }
        base = GradientBoostingClassifier(random_state=42)
    
    else:   # Random Forest
        param_dist = {
            'n_estimators':      [100, 200, 300, 400],
            'max_depth':         [None, 10, 20, 30],
            'max_features':      ['sqrt', 'log2', 0.3],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':  [1, 2, 4],
        }
    base = RandomForestClassifier(random_state=42, n_jobs=ARGS.n_jobs)
    
    # FIX: apply scaler when tuning Logistic Regression
    X_tune_tr = X_train_sc if tune_uses_scaler else X_train
    X_tune_te = X_test_sc  if tune_uses_scaler else X_test
    
    search = RandomizedSearchCV(
        base, param_dist,
        n_iter       = 30,
        cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring      = 'roc_auc',
        random_state = 42,
        n_jobs       = ARGS.n_jobs,
        verbose      = 1,
    )
    search.fit(X_tune_tr, y_train)
    
    print(f"\n  Best params : {search.best_params_}")
    print(f"  Best CV AUC : {search.best_score_:.4f}")
    
    best_tuned  = search.best_estimator_
    y_pred_t    = best_tuned.predict(X_tune_te)
    y_prob_t    = best_tuned.predict_proba(X_tune_te)[:, 1]
    
    tuned_results = {
        'Accuracy':  round(accuracy_score(y_test, y_pred_t),                   4),
        'Precision': round(precision_score(y_test, y_pred_t, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, y_pred_t,    zero_division=0), 4),
        'F1-Score':  round(f1_score(y_test, y_pred_t,        zero_division=0), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob_t),                    4),
    }
    print(f"\n  Tuned {tune_name} — Test Metrics:")
    for k, v in tuned_results.items():
        diff  = v - results[tune_name][k]
        arrow = '↑' if diff > 0 else ('↓' if diff < 0 else '─')
        print(f"    {k:<12}: {v:.4f}  {arrow}{abs(diff):.4f}")
    
    # Save final model
    if tuned_results['ROC-AUC'] >= results[tune_name]['ROC-AUC']:
        final_model = best_tuned
        final_name  = f"Tuned {tune_name}"
    else:
        final_model = fitted[tune_name]
        final_name  = tune_name

joblib.dump(final_model, output_path('badminton_best_model.pkl'))
print(f"\n  Final model: {final_name}")
print("  Saved: outputs/badminton_best_model.pkl")

# =============================================================================
# DONE — Summary
# =============================================================================
print("\n" + "=" * 70)
print("  PIPELINE COMPLETE — all outputs in ./outputs/")
print("=" * 70)
print(f"""
  Dataset
  ─────────────────────────────────────────────────────────────────
  Source        : BWF World Tour (sanderp/badminton-bwf-world-tour)
  Raw rows      : 3,761
  After cleaning: {model_df.shape[0]:,}  (retirements & NaN rows removed)
  Features used : {len(FEATURES)}
  Target        : team_one_wins (binary: 1=T1 wins, 0=T2 wins)
  Train/Test    : 80% / 20%  (stratified)

  Model Results (test set)
  ─────────────────────────────────────────────────────────────────
""")
for name, r in results.items():
    best_marker = ' ← BEST' if name == results_df['ROC-AUC'].idxmax() else ''
    print(f"  {name:<25} Acc={r['Accuracy']}  F1={r['F1-Score']}  AUC={r['ROC-AUC']}{best_marker}")

print(f"""
  Output files
  ─────────────────────────────────────────────────────────────────
  eda_overview.png                class dist + correlation heatmap
  eda_missing_values.png          missing value chart
  eda_feature_distributions.png   differential feature histograms
  eda_boxplot_total_points.png    boxplot — strongest feature
  model_comparison.csv            all metric scores
  model_comparison_barplot.png    grouped bar chart
  roc_curves.png                  ROC curves for all models
  confusion_matrices.png          all confusion matrices
  learning_curves.png             bias/variance learning curves
  feature_importance_rf.png       RF Gini importances
  cv_scores.csv                   5-fold CV mean ± std
  calibration_curve.png           calibration curves (isotonic)
  shap_summary_bar.png            SHAP global bar chart
  shap_summary_beeswarm.png       SHAP beeswarm
  shap_beeswarm_clean.png         SHAP beeswarm top-10 (clean)
  shap_interactions.png           SHAP interaction effects (tree only)
  shap_dependence_<feat>.png      SHAP dependence (top feature)
  shap_local_win_case.png         SHAP force plot — win case
  shap_local_loss_case.png        SHAP force plot — loss case
  shap_waterfall_win.png          SHAP waterfall — win case
  shap_waterfall_borderline.png   SHAP waterfall — borderline case
  lime_win_case.png + .html       LIME local — win case
  lime_loss_case.png + .html      LIME local — loss case
  badminton_best_model.pkl        serialised final model
  scaler.pkl                      serialised StandardScaler
  selected_features.txt           feature list
  ─────────────────────────────────────────────────────────────────
""")
