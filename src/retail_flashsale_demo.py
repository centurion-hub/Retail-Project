import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === 1. Load demo data ===
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "demo_flashion_data.csv"
df = pd.read_csv(DATA_FILE)

# === 2. Feature engineering ===
df["sell_through"] = df["Quantity Sold"] / df["Starting Inventory"]
df["is_sold_out"] = df["Quantity Sold"] == df["Starting Inventory"]
df["Price_to_MSRP"] = df["Price"] / df["MSRP"]
df["Gross_Margin_Rate"] = (df["Price"] - df["Cost"]) / df["Price"]

# === 3. Build cluster key for truncated demand recovery ===
df["cluster"] = (
    df["Department"].astype(str) + "_" +
    pd.qcut(df["Price_to_MSRP"], 4, labels=False).astype(str) + "_" +
    pd.qcut(df["Gross_Margin_Rate"], 4, labels=False).astype(str)
)

# === 4. Correct sell-through for sold-out items ===
df["sell_through_corrected"] = df["sell_through"]
cluster_estimates = {}

for name, group in df.groupby("cluster"):
    not_sold_out = group[group["is_sold_out"] == False]
    if len(not_sold_out) >= 10:
        est = max(
            not_sold_out["sell_through"].median(),
            not_sold_out["sell_through"].quantile(0.9)
        )
        cluster_estimates[name] = est

for idx, row in df[df["is_sold_out"]].iterrows():
    cluster_key = row["cluster"]
    if cluster_key in cluster_estimates:
        df.at[idx, "sell_through_corrected"] = cluster_estimates[cluster_key]

# === 5. Prepare features and target ===
features = ["Price", "Cost", "MSRP", "Department", "Month", "Event Start DOW", "Shipping Method"]
X = df[features]
y = df["sell_through_corrected"]

# === 6. Preprocessing (One-Hot Encoding) ===
categorical_features = ["Department", "Month", "Event Start DOW", "Shipping Method"]
numeric_features = ["Price", "Cost", "MSRP"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ("num", "passthrough", numeric_features)
])

X_encoded = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()
X_encoded = pd.DataFrame(X_encoded, columns=feature_names)

# === 7. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# === 8. Train LightGBM model ===
lgb_model = lgb.LGBMRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
)
lgb_model.fit(X_train, y_train)

# === 9. Model evaluation ===
y_pred = lgb_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# === 10. Visualization ===
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.xlabel("True Corrected Sell-Through")
plt.ylabel("Predicted Sell-Through (LightGBM)")
plt.title(f"LightGBM Prediction (Corrected Target)\nRMSE={rmse:.4f}, R²={r2:.4f}")
plt.grid(True)

# Save instead of just showing
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
out_path = RESULTS_DIR / "prediction_scatter.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Done. RMSE={rmse:.4f}, R²={r2:.4f}. Figure saved to {out_path}")
