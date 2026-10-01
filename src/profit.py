"""Profit layer: turn repeat-purchase probabilities into a contact-list decision.

Expected profit of contacting customer i (all money in BRL; every parameter is an ASSUMPTION):
    profit_i = p_i * uplift * margin * AOV          # extra margin from purchases the offer causes
             - p_i * (1 + uplift) * voucher          # voucher cost paid on every redemption (incl. 'sure things')
             - contact_cost                          # cost of reaching the customer
p_i is the model's calibrated probability of a repeat purchase; `uplift` is the assumed relative
increase in that probability caused by the contact (the model itself cannot measure this).
"""
import sqlite3, itertools
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ---- 1. Calibrated probabilities (no class weighting, so probabilities stay realistic) ----
tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y, idc = "repeat_purchase", "customer_unique_id"
num = [c for c in tr.columns if c not in (y, idc, "state")]
top = tr["state"].value_counts().head(10).index
for d in (tr, te): d["state"] = d["state"].where(d["state"].isin(top), "OTHER")
pre = ColumnTransformer([
    ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["state"])])
model = Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1000, C=0.1))])
model.fit(tr.drop(columns=[y, idc]), tr[y])
te["p"] = model.predict_proba(te.drop(columns=[y, idc]))[:, 1]

te["decile"] = pd.qcut(te["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
calib = te.groupby("decile", observed=True).agg(customers=(y, "size"), mean_predicted=("p", "mean"), actual_rate=(y, "mean")).round(4)
calib.to_csv("reports/calibration_by_decile.csv"); print("Calibration (decile 1 = highest score):\n", calib.to_string(), "\n")

# ---- 2. Business assumptions (all labelled assumptions) ----
con = sqlite3.connect("data/olist.db")
AOV = pd.read_sql("""SELECT AVG(v) a FROM (SELECT SUM(p.payment_value) v FROM orders o JOIN order_payments p USING(order_id)
    WHERE o.order_status='delivered' AND o.order_purchase_timestamp>='2018-03-01' AND o.order_purchase_timestamp<'2018-09-01'
    GROUP BY o.order_id)""", con).iloc[0, 0]                  # measured from data
BASE = dict(margin=0.15, voucher=10.0, contact=0.10, uplift=0.30)    # ASSUMPTIONS
print(f"AOV measured from data: R${AOV:.0f}.  Base assumptions: {BASE}")

def profit_curve(p, margin, voucher, contact, uplift, aov=AOV):
    ev = p * uplift * margin * aov - p * (1 + uplift) * voucher - contact
    order = np.argsort(-p)                       # contact highest-propensity first
    return np.cumsum(ev[order])

def best(p, **kw):
    c = profit_curve(p, **kw); k = int(np.argmax(c)); return k + 1, c[k]

p = te["p"].values; N = len(p)
k, val = best(p, **BASE)
print("Base case: no profitable list (campaign should NOT run)" if val <= 0 else f"Base case: best list = top {k:,} customers ({k/N:.1%}), expected profit R${val:,.0f}")

# ---- 3. Chart: cumulative profit vs list size for several offer designs ----
scen = {"Voucher R$10 + SMS (base)": BASE,
        "Voucher R$5": {**BASE, "voucher": 5.0},
        "E-mail only, no voucher": {**BASE, "voucher": 0.0, "contact": 0.02, "uplift": 0.10},
        "Voucher R$10, uplift 100%": {**BASE, "uplift": 1.0}}
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, a in scen.items():
    ax.plot(np.arange(1, N + 1) / N * 100, profit_curve(p, **a), label=name)
ax.axhline(0, color="grey", lw=0.8); ax.set_xlabel("% of customers contacted (highest score first)")
ax.set_ylabel("Cumulative expected profit (R$)"); ax.set_title("Retention campaign: expected profit by list size (assumptions, not measured)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("reports/profit_curve.png", dpi=150)

# ---- 4. Scenario table + sensitivity grid ----
rows = []
for name, a in scen.items():
    kk, v = best(p, **a); rows.append((name, kk if v > 0 else 0, round(kk / N * 100, 1) if v > 0 else 0, round(max(v, 0))))
sc = pd.DataFrame(rows, columns=["scenario", "customers_to_contact", "pct_of_base", "expected_profit_BRL"])
sc.to_csv("reports/scenarios.csv", index=False); print("\n", sc.to_string(index=False))

grid = []
for up, vo in itertools.product([0.1, 0.3, 0.5, 1.0, 2.0], [0, 2, 5, 10]):
    kk, v = best(p, **{**BASE, "uplift": up, "voucher": float(vo)})
    grid.append((up, vo, round(max(v, 0)), round(kk / N * 100, 1) if v > 0 else 0))
g = pd.DataFrame(grid, columns=["uplift", "voucher_BRL", "best_profit_BRL", "pct_contacted"])
g.pivot(index="uplift", columns="voucher_BRL", values="best_profit_BRL").to_csv("reports/sensitivity_profit.csv")
print("\nBest expected profit (R$) by uplift (rows) and voucher cost (cols):\n", g.pivot(index="uplift", columns="voucher_BRL", values="best_profit_BRL").to_string())
