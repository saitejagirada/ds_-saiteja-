# Trading Behavior & Market Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![ML](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange)

## 📖 Intrinsic Overview
This project serves as a quantitative investigation into the behavioral finance aspects of cryptocurrency trading. By fusing **historical trading logs (Hyperliquid)** with the **Bitcoin Fear & Greed Index**, we aim to mathematically determine if market sentiment dictates trading performance.

Unlike standard EDA, this implementation applies **Machine Learning (XGBoost Regressors)** and **Statistical Filtering (10-90th percentile outlier removal)** to strip away noise and identify the true drivers of profitability—whether it be the asset choice, the market mood, or the sheer size of the position.

## 📂 Repository Structure
```text
ds_<your_name>/
├── run_analysis.py        # The standalone script to run the analysis in terminal
├── notebook_1.ipynb       # The Jupyter Notebook for interactive exploration
├── historical_data.csv    # (Required) Raw trading data
├── fear_greed_index.csv   # (Required) Market sentiment data
├── outputs/               # Auto-generated charts and graphs
└── README.md              # Documentation
```
```
# 1. Clone the repository
git clone https://github.com/<saitejagirada>/ds_-saiteja-.git
cd ds_<YOUR_NAME>

# 2. Install required libraries
pip install pandas matplotlib seaborn scikit-learn xgboost

# 3. Run the analysis script
python run_analysis.py
```
---
