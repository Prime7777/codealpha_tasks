# 📊 Unemployment Rate Analysis in India

A data analysis project that explores unemployment trends in India,
investigates the impact of **Covid-19** on unemployment rates, and
identifies key regional and seasonal patterns.

Built as part of the **CodeAlpha Data Science Internship**.

---

## 📌 Project Overview

This project analyzes unemployment data across 28 Indian states from
2019 to 2020, with a special focus on how the Covid-19 pandemic and
the national lockdown (March 25, 2020) affected employment patterns.

---

## 📁 Project Structure

```
UNEMPLOYMENT_PROJECT/
│
├── unemployment_analysis.py          # Main analysis script
├── Unemployment in India.csv         # Primary dataset (768 records)
├── Unemployment_Rate_upto_11_2020.csv # Covid period dataset (267 records)
│
├── chart1_monthly_trend.png          # Monthly unemployment trend
├── chart2_regional.png               # Regional unemployment by state
├── chart3_covid_impact.png           # Covid-19 impact comparison
├── chart4_rural_urban.png            # Rural vs Urban trend
└── chart5_top_bottom_states.png      # Best & worst performing states
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Prime7777/CodeAlpha_UnemploymentAnalysis.git
cd CodeAlpha_UnemploymentAnalysis
```

### 2. Install required libraries
```bash
pip install pandas numpy matplotlib
```

### 3. Run the analysis
```bash
python unemployment_analysis.py
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Overall Avg Unemployment** | 11.79% |
| **Pre-Covid Avg (Jan–Mar 2020)** | 9.76% |
| **During Covid Avg (Apr–Nov 2020)** | 13.28% |
| **Covid Spike** | +3.52% |
| **Highest State** | Haryana (27.48%) |
| **Lowest State** | Meghalaya (3.87%) |

---

## 📈 Visualizations

### Chart 1 — Monthly Unemployment Trend
Shows the unemployment rate month by month with the lockdown date
(March 25, 2020) marked in red. A sharp spike is visible after lockdown.

### Chart 2 — Regional Unemployment by State
Horizontal bar chart comparing all 28 states.
- 🔴 Highest: Haryana (27.48%)
- 🟢 Lowest: Meghalaya (3.87%)

### Chart 3 — Covid-19 Impact Comparison
Side by side comparison of Pre-Covid vs During-Covid unemployment rates.
Covid caused a **3.52% increase** in unemployment across India.

### Chart 4 — Rural vs Urban Trend
Compares how rural and urban areas were differently affected.
Urban areas saw a sharper spike during lockdown than rural areas.

### Chart 5 — Top 5 vs Bottom 5 States
Highlights the best and worst performing states in terms of
unemployment rates during 2020.

---

## 🔍 Key Insights

1. **Covid-19 Impact** — Unemployment jumped from 9.76% to 13.28% after lockdown
2. **Urban areas** were more severely affected than rural areas
3. **Haryana** had the highest unemployment rate at 27.48%
4. **Meghalaya** was the most stable state at only 3.87%
5. **April–May 2020** saw the peak unemployment during the strictest lockdown
6. Unemployment gradually recovered after June 2020 as lockdown eased

---

## 💡 Policy Insights

- Urban workforce needs stronger social safety nets during crises
- States like Haryana need targeted employment programs
- Rural employment schemes (like MGNREGA) helped cushion rural impact
- Future pandemic preparedness should include employment protection plans

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, and analysis |
| `numpy` | Numerical operations |
| `matplotlib` | Charts and visualizations |

---

## 🧠 Concepts Learned

- **Data Cleaning** — Stripping whitespace, converting dates, handling nulls
- **Exploratory Data Analysis (EDA)** — Statistics and pattern finding
- **Time Series Analysis** — Tracking trends over time
- **Covid Impact Analysis** — Before vs after comparison
- **Regional Analysis** — State-wise unemployment comparison
- **Data Visualization** — 5 different chart types

---

## 📂 Dataset

- **Source:** [Kaggle — Unemployment in India](https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india)
- **Records:** 768 (File 1) + 267 (File 2)
- **States Covered:** 28 Indian states
- **Time Period:** 2019 – November 2020

---

## 👤 Author

**Anubhav Malik**
CodeAlpha Data Science Internship
