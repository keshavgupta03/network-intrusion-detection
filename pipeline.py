"""
Network Intrusion Detection on NSL-KDD
Stages: load -> clean -> split -> preprocess -> train 6 models -> evaluate -> save figures + best model.

Run:  python pipeline.py
Needs: data/KDDTrain+.txt  (and optionally data/KDDTest+.txt for the unseen-attack test)
Switch TASK to "multiclass" for Normal/DoS/Probe/R2L/U2R classification.
"""
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, classification_report)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ----------------------------- CONFIG -----------------------------
DATA_DIR = Path(__file__).parent


def find_file(name):
    """Return the real file called `name`, even if it sits inside a same-named folder."""
    for p in DATA_DIR.rglob(name):
        if p.is_file():
            return p
    return DATA_DIR / name


TRAIN_FILE = find_file("KDDTrain+.txt")
TEST_FILE = find_file("KDDTest+.txt")
OUT = Path("outputs")
TASK = "binary"          # "binary" or "multiclass"
SEED = 42
CV_FOLDS = 3
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_srv_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label", "difficulty"]
CAT_COLS = ["protocol_type", "service", "flag"]

ATTACK_MAP = {}
for k, names in {
    "DoS": "back land neptune pod smurf teardrop apache2 udpstorm processtable worm mailbomb",
    "Probe": "ipsweep nmap portsweep satan mscan saint",
    "R2L": "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster sendmail named "
           "snmpgetattack snmpguess xlock xsnoop httptunnel",
    "U2R": "buffer_overflow loadmodule perl rootkit ps sqlattack xterm",
}.items():
    for n in names.split():
        ATTACK_MAP[n] = k
ATTACK_MAP["normal"] = "Normal"

CLASS_NAMES = ["normal", "attack"] if TASK == "binary" else ["Normal", "DoS", "Probe", "R2L", "U2R"]


# ----------------------------- 1. LOAD + CLEAN -----------------------------
def load_raw(path):
    return pd.read_csv(path, header=None, names=COLS)


def clean(df, report=False, drop_dupes=True):
    df = df.copy()
    if report:
        print(f"Raw shape: {df.shape}")
        print(f"Missing values: {int(df.isna().sum().sum())}")
        print(f"Duplicate rows: {int(df.duplicated().sum())}")
    df = df.drop(columns=["difficulty"])                      # not a network feature
    df["label"] = df["label"].str.strip().str.lower()
    if drop_dupes:
        df = df.drop_duplicates()
    df["category"] = df["label"].map(ATTACK_MAP)
    unmapped = df["category"].isna().sum()
    if unmapped:
        print(f"Dropping {unmapped} rows with unmapped attack names")
        df = df.dropna(subset=["category"])
    if TASK == "binary":
        df["target"] = (df["category"] != "Normal").astype(int)
    else:
        df["target"] = df["category"].map({c: i for i, c in enumerate(CLASS_NAMES)})
    return df.reset_index(drop=True)


raw = load_raw(TRAIN_FILE)
df = clean(raw, report=True)
const_cols = [c for c in df.columns if df[c].nunique() <= 1]
print(f"Constant columns dropped: {const_cols}")
df = df.drop(columns=const_cols)
print(f"Clean shape: {df.shape}")

FEATURES = [c for c in COLS if c in df.columns and c not in ("label", "difficulty")]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

# --- EDA figures (before preprocessing) ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df["category"].value_counts().plot(kind="bar", ax=ax[0], color=sns.color_palette("deep"))
ax[0].set_title("Class distribution (5 categories)"); ax[0].set_ylabel("Records")
df["protocol_type"].value_counts().plot(kind="bar", ax=ax[1], color=sns.color_palette("muted"))
ax[1].set_title("Protocol types"); ax[1].set_ylabel("Records")
plt.tight_layout(); plt.savefig(OUT / "fig_class_distribution.png", dpi=200); plt.close()

plt.figure(figsize=(12, 9))
sns.heatmap(df[NUM_COLS].corr(), cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .7})
plt.title("Correlation matrix of numeric features")
plt.tight_layout(); plt.savefig(OUT / "fig_correlation.png", dpi=200); plt.close()

# ----------------------------- 2. SPLIT (stratified 70/15/15) -----------------------------
X, y = df[FEATURES], df["target"]
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print(f"Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")


# ----------------------------- 3. PREPROCESSING + MODELS -----------------------------
def make_prep():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("num", StandardScaler(), NUM_COLS),
    ])


def models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.9,
                                 colsample_bytree=0.9, eval_metric="logloss", n_jobs=-1, random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63,
                                   class_weight="balanced", n_jobs=-1, random_state=SEED, verbose=-1),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=60, early_stopping=True,
                                          random_state=SEED),
    }


def evaluate(pipe, X_, y_):
    pred = pipe.predict(X_)
    proba = pipe.predict_proba(X_)
    avg = "binary" if TASK == "binary" else "weighted"
    try:
        auc = roc_auc_score(y_, proba[:, 1]) if TASK == "binary" else \
            roc_auc_score(y_, proba, multi_class="ovr", average="weighted", labels=list(range(len(CLASS_NAMES))))
    except ValueError:
        auc = np.nan
    return {"Accuracy": accuracy_score(y_, pred),
            "Precision": precision_score(y_, pred, average=avg, zero_division=0),
            "Recall": recall_score(y_, pred, average=avg, zero_division=0),
            "F1": f1_score(y_, pred, average=avg, zero_division=0),
            "ROC-AUC": auc}


# ----------------------------- 4. TRAIN + COMPARE -----------------------------
fitted, rows = {}, []
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
cv_scoring = "f1" if TASK == "binary" else "f1_weighted"
for name, clf in models().items():
    pipe = Pipeline([("prep", make_prep()), ("clf", clf)])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0
    cv_f1 = cross_val_score(Pipeline([("prep", make_prep()), ("clf", models()[name])]),
                            X_train, y_train, cv=cv, scoring=cv_scoring, n_jobs=1).mean()
    val_m = evaluate(pipe, X_val, y_val)
    test_m = evaluate(pipe, X_test, y_test)
    fitted[name] = pipe
    rows.append({"Model": name, **{k: round(v, 4) for k, v in test_m.items()},
                 f"CV F1 ({CV_FOLDS}-fold)": round(cv_f1, 4), "Val F1": round(val_m["F1"], 4),
                 "Train time (s)": round(train_time, 1)})
    print(rows[-1])

results = pd.DataFrame(rows).set_index("Model")
results.to_csv(OUT / "model_comparison_test.csv")
print("\n", results.to_string())

best_name = results["Val F1"].idxmax()          # selected on validation, reported on test
best_pipe = fitted[best_name]
print(f"\nBest model (by validation F1): {best_name}")

# ----------------------------- 5. FIGURES -----------------------------
metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
ax = results[metric_cols].plot(kind="bar", figsize=(11, 5), width=0.8)
ax.set_ylim(max(0, results[metric_cols].min().min() - 0.05), 1.01)
ax.set_title("Model comparison on held-out test set"); ax.set_ylabel("Score")
plt.xticks(rotation=25, ha="right"); plt.legend(loc="lower right", ncol=5, fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_model_comparison.png", dpi=200); plt.close()

if TASK == "binary":
    plt.figure(figsize=(7, 6))
    for name, pipe in fitted.items():
        fpr, tpr, _ = roc_curve(y_test, pipe.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=f"{name} (AUC={results.loc[name, 'ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (test set)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "fig_roc_curves.png", dpi=200); plt.close()

cm = confusion_matrix(y_test, best_pipe.predict(X_test), labels=list(range(len(CLASS_NAMES))))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout(); plt.savefig(OUT / "fig_confusion_matrix.png", dpi=200); plt.close()

with open(OUT / "classification_report_best.txt", "w") as f:
    f.write(classification_report(y_test, best_pipe.predict(X_test), target_names=CLASS_NAMES, zero_division=0))

rf = fitted["Random Forest"]
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values().tail(15)
plt.figure(figsize=(8, 6)); imp.plot(kind="barh", color="#2a6f97")
plt.title("Top 15 features (Random Forest importance)"); plt.xlabel("Importance")
plt.tight_layout(); plt.savefig(OUT / "fig_feature_importance.png", dpi=200); plt.close()

# ----------------------------- 6. UNSEEN-ATTACK TEST (KDDTest+) -----------------------------
if TEST_FILE.exists():
    ext = clean(load_raw(TEST_FILE), drop_dupes=False).drop(columns=const_cols, errors="ignore")
    ext_rows = []
    for name, pipe in fitted.items():
        m = evaluate(pipe, ext[FEATURES], ext["target"])
        ext_rows.append({"Model": name, **{k: round(v, 4) for k, v in m.items()}})
    ext_df = pd.DataFrame(ext_rows).set_index("Model")
    ext_df.to_csv(OUT / "model_comparison_kddtest_unseen.csv")
    print("\nKDDTest+ (contains attacks unseen in training):\n", ext_df.to_string())
else:
    print("\nKDDTest+.txt not found - skipping unseen-attack evaluation.")

# ----------------------------- 7. SAVE FOR THE APP -----------------------------
joblib.dump(best_pipe, OUT / "best_model.joblib")
meta = {"task": TASK, "classes": CLASS_NAMES, "features": FEATURES, "best_model": best_name,
        "categories": {c: sorted(df[c].unique().tolist()) for c in CAT_COLS},
        "test_metrics": results.loc[best_name, metric_cols].to_dict()}
with open(OUT / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

sample_idx = pd.concat([y_test[y_test == k].sample(min(8, (y_test == k).sum()), random_state=SEED)
                        for k in sorted(y_test.unique())]).index
samples = X_test.loc[sample_idx].copy()
samples["true_label"] = [CLASS_NAMES[i] for i in y_test.loc[sample_idx]]
samples.sample(frac=1, random_state=SEED).to_csv(OUT / "samples.csv", index=False)
print("\nDone. Outputs in ./outputs")
