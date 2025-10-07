# retail_flashsale_demo.py
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt


# ---------------------------
# Utilities
# ---------------------------
def parse_numeric(s: pd.Series) -> pd.Series:
    """
    Robust numeric parser: handles $, commas, %, and parentheses for negatives.
    Converts invalid strings to NaN.
    """
    if s.dtype != "object":
        return pd.to_numeric(s, errors="coerce")
    s = s.replace(r"^\s*$", pd.NA, regex=True)  # empty -> NaN
    # (123.45) -> -123.45
    s_clean = (
        s.str.replace(r"[\$,]", "", regex=True)
         .str.replace("%", "", regex=False)
         .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
         .str.replace(r"\s+", "", regex=True)
    )
    return pd.to_numeric(s_clean, errors="coerce")


def bin_edges_from_qcut(categorical_series: pd.Series) -> np.ndarray:
    """
    Extract numeric bin edges from a qcut-generated categorical Series with IntervalIndex categories.
    """
    cats = categorical_series.cat.categories  # IntervalIndex
    edges = [iv.left for iv in cats] + [cats[-1].right]
    return np.array(edges, dtype=float)


def apply_stockout_correction(df_in: pd.DataFrame,
                              price_edges: np.ndarray,
                              gm_edges: np.ndarray,
                              cluster_estimates: pd.Series) -> pd.DataFrame:
    """
    Apply stock-out (truncation) correction using TRAIN-derived bin edges and cluster estimates.
    - Build TEST/TRAIN bins with TRAIN edges (no fitting on df_in)
    - Create cluster key consistent with TRAIN
    - Replace sold-out rows' sell-through with cluster median from TRAIN
    """
    df = df_in.copy()

    gm = (df["Price"] - df["Cost"]) / df["Price"]
    price_bins = pd.cut(df["Price_to_MSRP"], bins=price_edges, include_lowest=True)
    gm_bins = pd.cut(gm, bins=gm_edges, include_lowest=True)

    df["cluster"] = (
        df["Department"].astype(str) + "_" +
        price_bins.astype(str) + "_" +
        gm_bins.astype(str)
    )

    df["is_sold_out"] = df["Quantity Sold"] >= df["Starting Inventory"]

    df["sell_through_corrected"] = df["sell_through"].copy()
    mask = df["is_sold_out"] & df["cluster"].isin(cluster_estimates.index)
    if mask.any():
        df.loc[mask, "sell_through_corrected"] = cluster_estimates.loc[
            df.loc[mask, "cluster"]
        ].values

    df["sell_through_corrected"] = df["sell_through_corrected"].clip(0, 1)
    return df


def safe_random_split_index(index_arr: np.ndarray, test_ratio: float = 0.2, seed: int = 42):
    """
    Safe random split that guarantees at least 1 row in train and test (when n >= 2).
    """
    n = len(index_arr)
    if n < 2:
        raise ValueError(f"Not enough rows to split (n={n}).")
    test_n = max(1, int(round(test_ratio * n)))
    train_n = n - test_n
    if train_n < 1:
        test_n = 1
        train_n = n - 1
    rng = np.random.default_rng(seed)
    shuffled = index_arr.copy()
    rng.shuffle(shuffled)
    test_idx = shuffled[:test_n]
    train_idx = shuffled[test_n:]
    return train_idx, test_idx


# ---------------------------
# Main
# ---------------------------
def main():
    # === 0) Paths ===
    data_path = Path(r"D:\Retail_Project\data\demo_flashion_data.csv")
    results_dir = Path(r"D:\Retail_Project\results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # === 1) Load data (robust parsing) ===
    # Read all as string first to avoid early type mis-inference
    df = pd.read_csv(data_path, dtype=str, keep_default_na=True)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Required columns
    num_cols = ["Price", "MSRP", "Cost", "Starting Inventory", "Quantity Sold"]
    cat_cols = ["Department", "Month", "Event Start DOW", "Shipping Method"]

    for c in num_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required numeric column: {c}")
        df[c] = parse_numeric(df[c])

    for c in cat_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required categorical column: {c}")
        df[c] = df[c].astype(str)

    # Basic validity filters
    df = df.dropna(subset=num_cols)
    df = df[(df["Starting Inventory"] > 0) & (df["Price"] > 0) & (df["MSRP"] > 0)]
    df = df[(df["Price"] <= df["MSRP"]) & (df["Cost"] <= df["Price"])]

    # === 2) Feature engineering ===
    df["sell_through"] = (df["Quantity Sold"] / df["Starting Inventory"]).clip(0, 1)
    df["Price_to_MSRP"] = (df["Price"] / df["MSRP"]).clip(lower=0)
    df["Gross_Margin_Rate"] = ((df["Price"] - df["Cost"]) / df["Price"]).clip(-1, 1)

    df = df.dropna(subset=["sell_through", "Price_to_MSRP", "Gross_Margin_Rate"])

    # Early sanity check
    n = len(df)
    if n < 5:
        raise ValueError(f"Not enough rows after cleaning: {n}. Please check the raw CSV and parsing rules.")

    # === 3) Split FIRST (avoid leakage) ===
    all_idx = df.index.to_numpy()
    train_idx, test_idx = safe_random_split_index(all_idx, test_ratio=0.2, seed=42)
    df_tr = df.loc[train_idx].copy()
    df_te = df.loc[test_idx].copy()

    print(f"[INFO] Cleaned rows: {n} | Train: {len(df_tr)} | Test: {len(df_te)}")
    print(f"[INFO] dtypes → Price={df['Price'].dtype}, MSRP={df['MSRP'].dtype}, Cost={df['Cost'].dtype}")

    # === 4) Binning & clusters (TRAIN only) ===
    # qcut on TRAIN; duplicates='drop' to avoid identical bin edges error
    price_bins_tr = pd.qcut(df_tr["Price_to_MSRP"], q=4, duplicates="drop")
    gm_bins_tr = pd.qcut(df_tr["Gross_Margin_Rate"], q=4, duplicates="drop")

    df_tr["cluster"] = (
        df_tr["Department"].astype(str) + "_" +
        price_bins_tr.astype(str) + "_" +
        gm_bins_tr.astype(str)
    )

    df_tr["is_sold_out"] = df_tr["Quantity Sold"] >= df_tr["Starting Inventory"]
    not_sold_out_tr = df_tr.loc[~df_tr["is_sold_out"]]

    # Cluster-level estimate: median of NON-sold-out sell_through (robust; bounded)
    cluster_est = (
        not_sold_out_tr.groupby("cluster")["sell_through"].median().clip(0, 1)
    )

    # Numeric bin edges (from TRAIN only)
    price_edges = bin_edges_from_qcut(price_bins_tr)
    gm_edges = bin_edges_from_qcut(gm_bins_tr)

    # === 5) Apply correction on TRAIN and TEST (TEST uses TRAIN edges & estimates) ===
    df_tr_corr = apply_stockout_correction(df_tr, price_edges, gm_edges, cluster_est)
    df_te_corr = apply_stockout_correction(df_te, price_edges, gm_edges, cluster_est)

    # === 6) Prepare features & target ===
    features = ["Price", "Cost", "MSRP", "Department", "Month", "Event Start DOW", "Shipping Method"]
    cat_feats = ["Department", "Month", "Event Start DOW", "Shipping Method"]
    num_feats = ["Price", "Cost", "MSRP"]

    for c in cat_feats:
        df_tr_corr[c] = df_tr_corr[c].astype(str)
        df_te_corr[c] = df_te_corr[c].astype(str)

    X_train = df_tr_corr[features].copy()
    y_train = df_tr_corr["sell_through_corrected"].clip(0, 1)
    X_test = df_te_corr[features].copy()
    y_test = df_te_corr["sell_through_corrected"].clip(0, 1)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
            ("num", "passthrough", num_feats),
        ]
    )

    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # === 7) Train LightGBM ===
    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(
    X_train_enc, y_train,
    eval_set=[(X_test_enc, y_test)],
    eval_metric="rmse"
    )

    # === 8) Evaluation ===
    y_pred = model.predict(X_test_enc)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE={rmse:.4f}, R²={r2:.4f}")

    # === 9) Feature importance ===
    try:
        importances = model.booster_.feature_importance(importance_type="gain")
        feat_names = preprocessor.get_feature_names_out()
        fi = (
            pd.DataFrame({"feature": feat_names, "gain": importances})
            .sort_values("gain", ascending=False)
        )
        fi_path = results_dir / "lgbm_feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        print(f"[INFO] Feature importance saved to {fi_path}")
    except Exception as e:
        print(f"[WARN] Feature importance export skipped: {e}")

    # === 10) Scatter plot ===
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.35, s=15)
    plt.xlabel("True Corrected Sell-Through")
    plt.ylabel("Predicted Sell-Through (LightGBM)")
    plt.title(f"LightGBM Prediction (No-Leakage)\nRMSE={rmse:.4f}, R²={r2:.4f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = results_dir / "prediction_scatter.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[INFO] Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
