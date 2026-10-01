"""Baseline + gradient boosting on a time-based split. Metrics suited to rare positives."""
import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y_col, id_col = "repeat_purchase", "customer_unique_id"
num = [c for c in tr.columns if c not in (y_col, id_col, "state")]
top_states = tr["state"].value_counts().head(10).index
for d in (tr, te):
    d["state"] = d["state"].where(d["state"].isin(top_states), "OTHER")

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["state"]),
])
models = {
    "LogReg": Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "HistGB": Pipeline([("pre", pre), ("m", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0))]),
}
base = te[y_col].mean()
print(f"Test base rate: {base:.2%}  (random PR-AUC ~ {base:.4f})")
rows = []
for name, m in models.items():
    m.fit(tr.drop(columns=[y_col, id_col]), tr[y_col])
    p = m.predict_proba(te.drop(columns=[y_col, id_col]))[:, 1]
    te[f"p_{name}"] = p
    order = np.argsort(-p); k = int(0.1 * len(te))
    cap = te[y_col].values[order][:k].sum() / te[y_col].sum()
    rows.append((name, roc_auc_score(te[y_col], p), average_precision_score(te[y_col], p), cap, cap / 0.1))
print(pd.DataFrame(rows, columns=["model", "ROC-AUC", "PR-AUC", "top-10% capture", "lift@10%"]).round(3).to_string(index=False))
te.to_csv("data/test_scored.csv", index=False)
