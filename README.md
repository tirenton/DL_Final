# GRU-based Model for Bitcoin Price Forecasting  
**แบบจำลอง GRU สำหรับคาดการณ์ราคา Bitcoin โดยใช้ข้อมูลมหภาคหลายสินทรัพย์และ Regime ของความผันผวน**

---

## 🎯 Project Overview

This project builds a **Gated Recurrent Unit (GRU)**–based deep learning model to forecast the **direction of Bitcoin (BTC)** over the next **30 days**, using:

- Multi-asset financial time series
- Technical indicators
- Volatility regimes (low / mid / high)

The goal is not to predict exact prices, but to answer a practical trading question:

> “In the near future, is the price more likely to go **up** or **not up** from here?”

:contentReference[oaicite:0]{index=0}

---

## 📂 Course & Context

- **Course:** 204466 การเรียนรู้เชิงลึก (Deep Learning)  
- **Institution:** Kasetsart University  
- **Project Type:** Final project – end-to-end forecasting system  
- **Framework:** Python + PyTorch

---

## 📊 Data & Features

### Data Source

All market data are downloaded via **`yfinance`** (Yahoo Finance):

- **BTC-USD** – Bitcoin  
- **^GSPC** – S&P 500 Index  
- **GC=F** – Gold Futures  
- **^IXIC** – NASDAQ Composite  

Time range: from **2018-01-01** up to the data download date.

### Prediction Target

Binary classification label:

- `y = 1` if **BTC price in 30 days** is higher than today  
- `y = 0` otherwise  

So each sample asks whether BTC will be **higher or not** after a 30-day horizon.

### Engineered Features (per day)

For each asset and for BTC specifically:

- **Log returns**  
- **Rolling volatility** (e.g. 14-day std of returns)  
- **RSI(14)**  
- **Moving averages**: MA7, MA21  
- **Momentum**: cumulative return over 10 days  
- **MACD & signal line**

Cross-asset relationships:

- **Rolling correlations** between BTC and:
  - S&P 500
  - Gold
  - NASDAQ

Volatility regimes for BTC:

- Compute **30-day realized volatility**
- Split into **3 regimes** via quantiles:
  - Low / Mid / High  
- Encode as **one-hot** features (regime-aware model)

All features are **standardized** with `StandardScaler`, fit only on the **training set** to avoid data leakage.

---

## 🧠 Model Architecture

The model is a **multi-layer GRU** for sequence modeling.

### Input Shape

- Sequence length: **128 days**
- Each time step: feature vector size **F** (number of engineered features)
- Batch input: `[batch_size, 128, F]`

### GRU Stack

- **Type:** `nn.GRU`
- **Hidden size (H):** `256`
- **Layers:** `3`
- **Dropout:** `0.5` (between GRU layers)

At each time step, the GRU maintains a **hidden state** of size 256. Internally it uses:

- **Update gate `z`** – controls how much new information overrides old memory  
- **Reset gate `r`** – controls how much of the past to forget  
- **Candidate hidden state** – proposed new memory  

Activation functions:

- **Sigmoid** for gates (`z`, `r`)  
- **Tanh** for candidate hidden state  

The **same GRU parameters** are reused across all time steps in a layer.

### Output Layer

1. Take the **last hidden state** of the top GRU layer:  
   - Shape: `[batch_size, 256]`
2. Apply **Dropout(0.5)**.
3. Pass through a **Linear layer**:
   - `Linear(256 → 1)` → single **logit** per sample

At evaluation time, apply **sigmoid** to the logit to get:

- `p = P(price_up_in_30_days | features)`

---

## 🏋️‍♀️ Training Setup

- **Loss:** `BCEWithLogitsLoss`
  - Combines sigmoid + binary cross-entropy in a numerically stable way
- **Optimizer:** `AdamW`
- **Learning rate schedule:** `CosineAnnealingLR`
- **Batch size:** `64`
- **Max epochs:** `200`
- **Early stopping:**  
  - Monitors **validation loss**  
  - Stops if no improvement for **10 epochs**  
  - Restores the **best** model weights

### Time-based Split

To respect time ordering and avoid look-ahead bias:

- Use a **chronological split** (e.g. ~80% train, ~20% validation)
- No shuffling across time when building sequences

### Sliding Window Construction

For each index `i`:

- Input sequence: days `[i, i+1, ..., i+127]`
- Label: day `i+128+H-1` (where H = 30 days ahead)

This creates supervised pairs `(sequence_128_days, direction_in_30_days)`.

---

## ✅ Evaluation

### Metrics

Used on the **validation set**:

- **Loss** – `BCEWithLogitsLoss`  
- **AUC (ROC-AUC)** – threshold-independent ranking ability  
- **F1 score** – harmonic mean of precision & recall (important for imbalanced labels)  
- **Accuracy** – overall correctness  

Because class balance is not exactly 50/50, **AUC + F1** are especially informative.

### Threshold Tuning

1. The model outputs probabilities `p`.
2. Default threshold = `0.5`.
3. Scan multiple thresholds (e.g. 0.2 → 0.8).
4. Select the threshold that **maximizes F1** on the validation set.
5. Report:
   - Best threshold
   - F1, Accuracy at this threshold
   - Confusion matrix at this threshold

### Example Best Model Result (30-day horizon)

At the chosen best threshold:

- **Confusion Matrix**  
  - TN = 106  
  - FP = 30  
  - FN = 17  
  - TP = 52  
- **Precision** ≈ 0.634  
- **Recall** ≈ 0.754  
- **F1** ≈ 0.689  
- **Accuracy** ≈ 0.771  
- **AUC** ≈ 0.800  

Interpretation:

- Good **discrimination** between up vs not-up regimes (AUC 0.80).  
- Reasonable balance between **catching true up-moves** (recall) and **avoiding false alarms** (precision).  
- Works **better on 30-day horizon** than very short-term prediction, where noise is higher.

---

## 🔍 Why Deep Learning (GRU)?

Compared to:

- **Rule-based technical strategies** – simple but rigid; hard to adapt dynamically.  
- **Linear models / ARIMA** – assume stationarity & linearity; often unrealistic in markets.  
- **Tree-based models (e.g. XGBoost)** – strong on tabular data but do not natively handle **sequence dynamics**.

GRU advantages:

- Learns complex **non-linear temporal patterns** end-to-end.  
- Handles **multi-asset sequences** naturally.  
- Can incorporate **regime features** and **cross-asset correlations** directly.

Trade-offs:

- Needs more data and regularization to avoid overfitting.  
- Harder to interpret than simple statistical models.  
- Requires more compute than classic models, but lighter than LSTM/Transformer in this setting.

---

## 📚 Related Work & Inspirations (High-level)

- Technical indicators: MA, MACD, RSI, momentum (e.g. Murphy – *Technical Analysis of the Financial Markets*).  
- Time-series momentum and returns predictability (e.g. Moskowitz et al.).  
- RNN family for financial time series: LSTM & GRU based models.  
- Volatility regimes & regime-switching models (e.g. Hamilton; Andersen et al. on realized volatility).  
- Changing cross-asset correlations under stress (e.g. Longin & Solnik).  
- Training practice: AdamW, cosine annealing, ROC/PR analysis, threshold tuning for binary classification.

---

## 👥 Authors & Contribution

> Fill in percentages as appropriate for the final report.

| Task                                      | อภิภู ชูเจริญประกิจ (6510503891) | อิทธิเดช นามเหลา (6510503905) |
| ----------------------------------------- | ---------------------------------- | -------------------------------- |
| Idea & problem design                     |                                    |                                  |
| Data collection & pipeline                |                                    |                                  |
| Feature engineering & regime design       |                                    |                                  |
| Model implementation (GRU) & training     |                                    |                                  |
| Evaluation & threshold analysis           |                                    |                                  |
| Visualization & result interpretation     |                                    |                                  |
| Report writing & documentation            |                                    |                                  |

---

## 📝 How to Use This Repo (Suggested Structure)

> Adjust to match your actual file layout.

```text
.
├── README.md              # This file
├── data/                  # (Optional) Cached CSV/Parquet data
├── notebooks/
│   └── exploration.ipynb  # EDA & quick experiments
├── src/
│   ├── dataset.py         # Window construction, scaling
│   ├── features.py        # Feature engineering (log-ret, RSI, MACD, regimes, etc.)
│   ├── model.py           # GRUModel definition
│   ├── train.py           # Training loop & early stopping
│   └── eval.py            # Metrics, threshold search, plots
└── requirements.txt       # Dependencies (PyTorch, yfinance, pandas, numpy, etc.)
