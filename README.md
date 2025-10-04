# Retail-FlashSale-Prediction

This project explores **demand prediction and price optimization** for flash-sale retailing (similar to platforms like Rue La La).  
The code demonstrates the full pipeline from **feature engineering** → **truncated demand correction** → **LightGBM modeling** → **evaluation and visualization**.

---

## Project Structure
```
Retail-FlashSale-Prediction/
│
├── data/
│   └── demo_flashion_data.csv   # Demo dataset (sampled)
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

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the demo script:
   ```bash
   cd src
   python retail_flashsale_demo.py
   ```

3. Results will be saved under `results/` (e.g., prediction_scatter.png).

---

## ⚠️ Disclaimer

This repository uses a **small demo dataset** (`demo_flashion_data.csv`) for reproducibility.  
The demo data does not fully represent the distribution of the full dataset, so the evaluation results  
(e.g., RMSE, R²) may differ significantly from those obtained with the actual data.  

The goal is to **showcase the modeling pipeline and implementation logic**, not to replicate production-level performance.  
Original business data cannot be shared due to confidentiality.

---
