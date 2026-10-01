"""Load Olist CSVs into SQLite, then build time-based train/test sets.

Usage: python src/build_dataset.py --raw data/raw --db data/olist.db
"""
import argparse, sqlite3, pathlib
import pandas as pd

TABLES = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
}
# (name, feature cutoff, label window end)
SPLITS = {
    "train": ("2017-09-01", "2018-03-01"),
    "test":  ("2018-03-01", "2018-09-01"),
}

def load(raw, db):
    con = sqlite3.connect(db)
    for t, f in TABLES.items():
        pd.read_csv(pathlib.Path(raw) / f).to_sql(t, con, if_exists="replace", index=False)
    return con

def build(con, cutoff, end):
    feats = pd.read_sql(open("sql/02_customer_features.sql").read(), con, params={"cutoff": cutoff})
    labels = pd.read_sql(open("sql/03_labels.sql").read(), con, params={"cutoff": cutoff, "end": end})
    df = feats.merge(labels, on="customer_unique_id", how="left")
    df["repeat_purchase"] = df["repeat_purchase"].fillna(0).astype(int)
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--db", default="data/olist.db")
    a = ap.parse_args()
    pathlib.Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    con = load(a.raw, a.db)
    for name, (cut, end) in SPLITS.items():
        d = build(con, cut, end)
        d.to_csv(f"data/{name}.csv", index=False)
        print(f"{name}: {len(d):,} customers, {d.repeat_purchase.sum():,} repeat buyers ({d.repeat_purchase.mean():.2%})")
