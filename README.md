# Retail-FlashSale-Prediction

This project explores **demand prediction and price optimization** for flash-sale retailing (similar to platforms like Rue La La).
The repo demonstrates a full pipeline from **feature engineering** → **truncated demand correction** → **LightGBM modeling** → **evaluation and visualization**.

---

## Dataset (Demo from Original File)

- The file `data/demo_flashion_data_from_original.csv` contains the **first 1,000 rows** extracted from the original dataset 
  `Flashion Art Science in Fashion Retailing.xlsx`.
- This subset is provided **solely for demonstration and testing**.
- It maintains the same column structure and semantics as the full dataset but does **not represent the complete data**.
- Any performance metrics (e.g., RMSE, R²) reported using this demo dataset should **not be interpreted as final production performance**.

**Columns**
- `Department`, `Month`, `Event Start DOW`, `Shipping Method`: categorical descriptors
- `MSRP`, `Price`, `Cost`: pricing-related features
- `Starting Inventory`, `Quantity Sold`: inventory & realized sales
- `is_sold_out`: whether the item sold out (truncation indicator)

---

## Project Structure
```
Retail-FlashSale-Prediction/
│
├── data/
│   ├── Flashion Art Science in Fashion Retailing.xlsx   # Original dataset
│   └── demo_flashion_data_from_original.csv             # 1,000-row demo subset
│
├── src/
│   └── retail_flashsale_demo.py     # End-to-end pipeline script
│
├── results/                         # Outputs (plots, metrics)
│
├── requirements.txt
└── README.md
```

---

## How to Run

1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Run the demo pipeline:
```bash
python src/retail_flashsale_demo.py
```

3) Outputs (plots, metrics) are saved to `results/`.

---