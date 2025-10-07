# Retail-FlashSale-Prediction

This project explores **demand prediction and price optimization** for flash-sale retailing (similar to platforms like Rue La La).
The repo demonstrates a full pipeline from **feature engineering** → **truncated demand correction** → **LightGBM modeling** → **evaluation and visualization**.

---

## Dataset (Demo from Original File)

- The file `data/demo_flashion_data.csv` contains the **first 1,000 rows** extracted from the original dataset.
- This subset is provided **solely for demonstration and testing**.
- It maintains the same column structure and semantics as the full dataset but does **not represent the complete data**.
- Any performance metrics (e.g., RMSE, R²) reported using this demo dataset should **not be interpreted as final production performance**.

**Columns**
The following columns are included:

| Column | Description |
|---------|-------------|
| **Style#** | Unique product identifier for each SKU or fashion style |
| **Event#** | Identifier of the flash-sale event the product belongs to |
| **Starting Inventory** | Total inventory available at the start of the event |
| **Quantity Sold** | Units sold during the event duration (typically 24 hours) |
| **Cost** | Per-unit acquisition cost paid by Flashion to the designer |
| **Price** | Per-unit selling price offered to customers during the event |
| **MSRP** | Per unit Manufacturer's (Designer's) Suggested Retail Price |
| **Department** | Flashion’s internal top-level product classification |
| **Month** | Month when the event started (Oct = October, Nov = November, Dec = December) |
| **Event Start Time** | Time of day when the flash-sale event began |
| **Event Start DOW** | Day of week (e.g., Monday–Sunday) when the event started |
| **Shipping Method** | “F” = Flashion owns inventory and ships to customer; “D” = designer ships directly (Flashion does not hold stock) |

---

## Project Structure
```
Retail_Project/
│
├── data/
│   └── demo_flashion_data.csv             # 1,000-row demo subset
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

3) Outputs (plots, metrics) are saved to `results/`. # All output files are saved under results/ and are automatically generated when you run the scripts.

---