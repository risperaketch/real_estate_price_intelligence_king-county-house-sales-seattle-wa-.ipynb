# Real_Estate_Price_Intelligence_King-County-House-Sales-Seattle-WA
### Low-Level TensorFlow · Architecture Progression · Batch Training | King County House Sales

**Skills demonstrated:** Python · TensorFlow (low-level API) · Neural Networks · Deep Learning · Regression Modeling · Feature Engineering · Log Transformation · Mini-Batch Training · Data Visualization · Real Estate Analytics · Business Decision-Making

---

## Project Overview

Built three progressively sophisticated neural networks **entirely from scratch using low-level TensorFlow** — manually initializing `tf.Variable` weight tensors, writing explicit forward passes with `tf.matmul`, computing gradients via `tf.GradientTape`, and applying optimizer updates directly — to predict house sale prices across 21,613 King County, Seattle residential transactions. Each model introduces one new architectural or training concept, culminating in a production-style mini-batch pipeline that reduces prediction error by **~28% over the baseline** and mirrors how large-scale real estate valuation systems are actually trained.

---

## Business Problem

Accurate house price prediction powers three high-value business functions — and errors carry direct financial consequences at each stage:

| Business Function | Financial Stake | Impact of Misprediction |
|---|---|---|
| **Mortgage underwriting** | Loan approval based on collateral value | Overvalued properties increase default exposure for lenders |
| **Automated Valuation Models (AVMs)** | Powers Zillow, Redfin, bank appraisal tools | $50K+ errors erode buyer/seller trust and platform revenue |
| **Investment portfolio screening** | REITs and institutions rank assets by return potential | Inaccurate valuations misrank properties and distort capital allocation |

**Solution:** A neural network regression model trained on structural property features that is retrain-friendly as new transaction data arrives — and architecture-ready to scale from 21K records to a national property database without code changes.

---

## Data Description

- **Dataset:** King County House Sales — 21,613 residential transactions, Seattle WA (May 2014 – May 2015)
- **Source:** [Kaggle — House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- **Target variable:** Sale price in USD ($75,000 – $7,700,000; median ~$450,000)
- **Features used:** `sqft_living`, `bedrooms`, `bathrooms`, `floors`, `waterfront`
- **Split:** 80% train (17,290 records) · 20% test (4,323 records), stratified random split
- **Target transformation:** `log(price)` applied before training — compresses the right-skewed distribution into near-normal form; predictions exponentiated back to dollars for evaluation
- **Feature scaling:** Min-Max normalization [0, 1] — prevents high-range features (sqft: 290–13,540) from dominating low-range features (waterfront: 0–1) during gradient updates

---

## Methodology

Three models built in sequence, each adding one concept while holding architecture constant where possible:

- **Exploratory Data Analysis (Section 1):** 12-panel dashboard — raw price distribution with mean/median lines, log-price distribution showing why log-transform is the correct target, scatter plots of all 4 numeric features vs price with trend lines, feature correlation heatmap, waterfront premium box plot, and price-tier breakdown (Entry / Mid / Premium / Luxury)
- **Preprocessing (Section 2):** Min-Max scaling visualized before/after; log-transform rationale explained quantitatively; TF constant tensors prepared for all three models
- **Model 1 — Baseline (1 hidden layer, 3 neurons):** 5→3→1 architecture; `tf.Variable` weights initialized with `random_normal`; ReLU activation; MAE loss; Adam optimizer; 1,000 full-batch iterations; gradients computed with `tf.GradientTape`
- **Model 2 — Deeper (2 hidden layers, 5→2 neurons):** 5→5→2→1 architecture; second hidden layer enables compound feature interaction learning; same full-batch training; direct comparison reveals the depth benefit
- **Model 3 — Production Batch (2 hidden layers, 5→2 neurons):** Identical to Model 2 in architecture; training loop shuffles indices and processes data in 1,000-sample mini-batches per epoch for 1,000 epochs — mirrors production pipelines that handle datasets too large for RAM
- **Evaluation:** MAE, RMSE, and MAPE on held-out test set; per-price-tier error breakdown; actual-vs-predicted scatter plots; overlapping residual distribution histograms; training loss convergence comparison across all three models

---

## Results

| Model | Architecture | Training Method | Test MAE | MAPE | Improvement |
|---|---|---|---|---|---|
| Model 1 — Baseline | 5 → 3 → 1 | Full-batch · 1,000 iter | ~$180K | ~35% | — |
| Model 2 — Deeper | 5 → 5 → 2 → 1 | Full-batch · 1,000 iter | ~$145K | ~28% | −19% |
| **Model 3 — Batch** | **5 → 5 → 2 → 1** | **Mini-batch · 1K samples · 1K epochs** | **~$130K** | **~25%** | **−28% ✅** |

**Per-segment breakdown — Model 3 (recommended):**

| Segment | Properties | Test MAE | % of Median | Business Verdict |
|---|---|---|---|---|
| Entry-level (<$300K) | ~3,800 | ~$80K | ~35% | Use for market trend monitoring; too wide for loan approval |
| Mid-range ($300K–$600K) | ~10,200 | ~$110K | ~25% | Acceptable for investment screening and pricing dashboards |
| Premium ($600K–$1M) | ~5,400 | ~$145K | ~20% | Directional valuation tool; not for transaction-level decisions |
| Luxury (>$1M) | ~2,200 | ~$300K+ | ~25%+ | Requires richer features — grade, condition, ZIP code |

**Key technical insight:** Mini-batch training (Model 3) outperforms full-batch (Model 2) despite identical architecture. The per-batch gradient noise acts as an implicit regularizer — a finding that transfers directly to any production retraining pipeline.

---

## Business Recommendations

| Use Case | Recommended Model | Rationale |
|---|---|---|
| Single-listing appraisal or quick valuation | Model 3 — Batch-Trained | Lowest MAE; inference in milliseconds |
| Portfolio risk screening (1,000+ properties) | Model 3 — Batch-Trained | Batch architecture scales to large property databases without RAM constraints |
| Rapid prototype or proof-of-concept | Model 1 — Baseline | Fewest parameters; easiest to audit and explain to non-technical stakeholders |
| Luxury segment (>$1M) | Requires feature enhancement | All models underperform — add `grade`, `condition`, `yr_built`, `zipcode` |

**Priority next step for production readiness:** Retrain Model 3 with `sqft_lot`, `grade`, `yr_built`, and ZIP code cluster encoding to target MAE below $80K across all price segments — the threshold accepted for mortgage underwriting AVM applications.

---

## Tools & Technologies

- **Language:** Python 3.10
- **Deep Learning:** TensorFlow 2.x — low-level API (`tf.Variable`, `tf.GradientTape`, `tf.matmul`, `tf.nn.relu`)
- **Data Processing:** Pandas · NumPy · Scikit-learn (MinMaxScaler, train_test_split)
- **Visualization:** Matplotlib · Seaborn
- **Model Persistence:** Pickle (weight dictionaries as `.pkl`)
- **Environment:** Google Colab · Jupyter Notebook

---

## Project Files

```
real-estate-price-intelligence/
│
├── Real_Estate_Price_Intelligence_King_County_House_Sales.ipynb  ← Full notebook (8 sections)
├── README.md                                                      ← This file
│
└── Generated outputs (produced when notebook runs end-to-end)
    │
    ├── eda_overview.png              ← 12-panel EDA: price distribution, log-price,
    │                                    4× feature-vs-price scatters with trend lines,
    │                                    correlation heatmap, waterfront premium box plot,
    │                                    price-tier bar chart
    │
    ├── preprocessing.png             ← Feature scaling before vs after Min-Max:
    │                                    side-by-side box plots showing normalization effect
    │
    ├── model_comparison.png          ← 3-row dashboard: training loss convergence
    │                                    (all 3 models), actual-vs-predicted scatter
    │                                    (1,000 test samples per model), MAE bar chart,
    │                                    overlapping residual distribution histograms
    │
    ├── model1_baseline_weights.pkl   ← Saved Model 1 weights (W1, b1, W2, b2)
    ├── model2_deeper_weights.pkl     ← Saved Model 2 weights (W1–W3, b1–b3)
    ├── model3_batch_weights.pkl      ← Saved Model 3 weights (W1–W3, b1–b3)
    │
    ├── model_predictions.csv         ← Test set: actual prices + all 3 model predictions
    │                                    + price tier labels for segment analysis
    ├── model_comparison.csv          ← MAE, RMSE, MAPE, parameters for all 3 models
    └── training_loss_history.csv     ← Per-iteration/epoch MAE loss for all 3 models
```

---

## How to Run

**Google Colab** *(recommended — GPU optional, CPU is sufficient)*

1. Upload the notebook at [colab.research.google.com](https://colab.research.google.com)
2. Upload `kc_house_data.csv` via the Colab file browser — or skip this step and the notebook runs on a built-in synthetic fallback automatically
3. Runtime → Run all

**Local Jupyter**
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Real_Estate_Price_Intelligence_King_County_House_Sales.ipynb
```

> **Dataset:** Download `kc_house_data.csv` from [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction). If the file is absent, the notebook generates representative synthetic data and runs without interruption.

---

## Author

**Aketch Okoth** · M.S. Business Analytics · Montclair State University  
*Actively seeking Data Analyst and Business Analyst roles in the United States.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?style=flat&logo=github&logoColor=white)](https://github.com/your-username)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat&logo=gmail&logoColor=white)](mailto:your-email@email.com)
