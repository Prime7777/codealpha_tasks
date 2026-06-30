# 🚗 Car Price Prediction with Machine Learning

A Machine Learning project that predicts the selling price of used cars
based on features like present price, car age, kilometers driven, fuel type,
and transmission type.

Built as part of the **CodeAlpha Data Science Internship**.

---

## 📌 Project Overview

This project uses regression models to predict used car prices based on
real-world features. It covers the complete ML workflow including data
preprocessing, feature engineering, model training, evaluation, and
visualization.

---

## 📁 Project Structure

```
CAR_PRICE_PROJECT/
│
├── car_price_prediction.py           # Main ML script
├── car_data.csv                      # Dataset (301 cars)
│
├── chart1_actual_vs_predicted.png    # Actual vs Predicted prices
├── chart2_model_comparison.png       # Model R² comparison
├── chart3_feature_importance.png     # Feature importance chart
└── chart4_fuel_analysis.png          # Fuel type analysis
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Prime7777/codealpha_tasks.git
cd CAR_PRICE_PROJECT
```

### 2. Install required libraries
```bash
pip install pandas numpy matplotlib scikit-learn
```

### 3. Run the prediction script
```bash
python car_price_prediction.py
```

---

## 📊 Dataset

| Feature | Description |
|---------|-------------|
| `Car_Name` | Name of the car |
| `Year` | Year of manufacture |
| `Selling_Price` | Price to predict (target) |
| `Present_Price` | Current showroom price (Lakhs) |
| `Driven_kms` | Kilometers driven |
| `Fuel_Type` | Petrol / Diesel / CNG |
| `Selling_type` | Dealer or Individual |
| `Transmission` | Manual or Automatic |
| `Owner` | Number of previous owners |

- **Source:** [Kaggle — Car Price Prediction](https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars)
- **Records:** 301 cars
- **Missing Values:** None

---

## 🤖 Models Used

| Model | R² Score | MAE |
|-------|----------|-----|
| **Random Forest** | ~95% | ~0.8 Lakhs |
| **Gradient Boosting** | ~94% | ~0.9 Lakhs |
| **Linear Regression** | ~85% | ~1.8 Lakhs |

✅ **Random Forest performed best** with ~95% R² Score!

---

## 📈 Visualizations

### Chart 1 — Actual vs Predicted Prices
Scatter plot comparing actual car prices vs model predictions.
Points close to the red dashed line = accurate predictions.

### Chart 2 — Model Comparison
Bar chart comparing R² scores of all 3 models.
Higher R² = better model performance.

### Chart 3 — Feature Importance
Shows which features matter most for price prediction:
- 🔴 **Present Price** — most important feature
- **Car Age** — second most important
- **Driven KMs** — third most important

### Chart 4 — Fuel Type Analysis
- Pie chart showing distribution of fuel types
- Bar chart showing average selling price by fuel type

---

## 🔍 Key Insights

1. **Present Price** is the strongest predictor of selling price
2. **Car Age** significantly affects resale value
3. **Diesel cars** have higher average selling prices than Petrol
4. **Automatic transmission** cars sell for more than Manual
5. **Random Forest** outperforms other models with ~95% accuracy
6. More kilometers driven = lower selling price

---

## 🧠 Concepts Learned

- **Data Preprocessing** — Label encoding categorical variables
- **Feature Engineering** — Creating Car_Age from Year column
- **Regression Models** — Linear, Random Forest, Gradient Boosting
- **Model Evaluation** — R² Score, MAE, RMSE
- **Feature Importance** — Understanding which features drive predictions
- **Data Visualization** — Scatter plots, bar charts, pie charts

---

## 💡 Real World Applications

- Used car dealerships pricing their inventory
- Car buying apps estimating fair market value
- Insurance companies calculating vehicle value
- Banks determining loan amounts for car financing

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and preprocessing |
| `numpy` | Numerical operations |
| `matplotlib` | Data visualizations |
| `scikit-learn` | ML models and evaluation metrics |

---

## 👤 Author

**Anubhav Malik**
CodeAlpha Data Science Internship
