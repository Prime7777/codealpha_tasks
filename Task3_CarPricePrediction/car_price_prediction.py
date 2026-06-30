# ── CAR PRICE PREDICTION ─────────────────────────────────────────
# Dataset: Used Cars dataset from Kaggle
# Goal: Predict car selling price using regression models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── STEP 1: Load Data ────────────────────────────────────────────
print("=" * 55)
print("  CAR PRICE PREDICTION")
print("=" * 55)

df = pd.read_csv('car_data.csv')
print(f"\nDataset loaded: {df.shape[0]} cars, {df.shape[1]} features")
print(f"\nFirst 5 rows:")
print(df.head())

# ── STEP 2: Data Preprocessing ───────────────────────────────────
print("\n--- Data Preprocessing ---")

# Create car age feature from year
df['Car_Age'] = 2024 - df['Year']

# Encode categorical columns
le = LabelEncoder()
df['Fuel_Type_enc']    = le.fit_transform(df['Fuel_Type'])
df['Selling_type_enc'] = le.fit_transform(df['Selling_type'])
df['Transmission_enc'] = le.fit_transform(df['Transmission'])

print(f"Fuel types:     {df['Fuel_Type'].unique()}")
print(f"Selling types:  {df['Selling_type'].unique()}")
print(f"Transmissions:  {df['Transmission'].unique()}")

# ── STEP 3: Feature Engineering ──────────────────────────────────
# Select features for model
features = ['Present_Price', 'Car_Age', 'Driven_kms',
            'Fuel_Type_enc', 'Selling_type_enc',
            'Transmission_enc', 'Owner']

X = df[features]
y = df['Selling_Price']

print(f"\nFeatures used: {features}")
print(f"Target: Selling_Price")

# ── STEP 4: Train/Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── STEP 5: Train Models ─────────────────────────────────────────
models = {
    'Linear Regression'     : LinearRegression(),
    'Random Forest'         : RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting'     : GradientBoostingRegressor(n_estimators=100, random_state=42),
}

results = {}
print("\n--- Model Results ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'pred': y_pred}
    print(f"\n{name}:")
    print(f"  R² Score : {r2:.4f} ({r2*100:.2f}%)")
    print(f"  MAE      : {mae:.4f} lakhs")
    print(f"  RMSE     : {rmse:.4f} lakhs")

# Best model
best = max(results, key=lambda x: results[x]['r2'])
print(f"\n★ Best Model: {best} (R² = {results[best]['r2']*100:.2f}%)")

# ════════════════════════════════════════════════════════
# CHART 1: Actual vs Predicted Prices
# ════════════════════════════════════════════════════════
print("\nShowing Chart 1 of 4 - Actual vs Predicted...")
fig, ax = plt.subplots(figsize=(9, 6))
y_pred_best = results[best]['pred']
ax.scatter(y_test, y_pred_best, color='#3498db', alpha=0.6, s=60)
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Perfect Prediction')
ax.set_title(f'Chart 1 — Actual vs Predicted Car Prices ({best})',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Actual Price (Lakhs)', fontsize=12)
ax.set_ylabel('Predicted Price (Lakhs)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('chart1_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()

# ════════════════════════════════════════════════════════
# CHART 2: Model Comparison (R² Score)
# ════════════════════════════════════════════════════════
print("Showing Chart 2 of 4 - Model Comparison...")
fig, ax = plt.subplots(figsize=(9, 6))
names  = list(results.keys())
r2s    = [results[n]['r2'] * 100 for n in names]
colors = ['#e74c3c' if n == best else '#3498db' for n in names]
bars   = ax.bar(names, r2s, color=colors, edgecolor='white', width=0.5)
ax.set_title('Chart 2 — Model Comparison (R² Score)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('R² Score (%)', fontsize=12)
ax.set_ylim(0, 115)
for bar, val in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1.5,
            f'{val:.2f}%', ha='center',
            fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('chart2_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ════════════════════════════════════════════════════════
# CHART 3: Feature Importance (Random Forest)
# ════════════════════════════════════════════════════════
print("Showing Chart 3 of 4 - Feature Importance...")
rf    = models['Random Forest']
imp   = pd.Series(rf.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#e74c3c' if v == imp.max() else '#3498db' for v in imp.values]
imp.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Chart 3 — Feature Importance (Random Forest)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Importance Score', fontsize=12)
for i, val in enumerate(imp.values):
    ax.text(val + 0.002, i, f'{val:.3f}',
            va='center', fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('chart3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ════════════════════════════════════════════════════════
# CHART 4: Price Distribution by Fuel Type
# ════════════════════════════════════════════════════════
print("Showing Chart 4 of 4 - Price by Fuel Type...")
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Fuel type distribution
fuel_counts = df['Fuel_Type'].value_counts()
axes[0].pie(fuel_counts.values,
            labels=fuel_counts.index,
            autopct='%1.1f%%',
            colors=['#3498db','#e74c3c','#2ecc71'],
            startangle=90)
axes[0].set_title('Car Count by Fuel Type',
                  fontsize=13, fontweight='bold')

# Avg price by fuel type
fuel_price = df.groupby('Fuel_Type')['Selling_Price'].mean().sort_values()
axes[1].bar(fuel_price.index, fuel_price.values,
            color=['#3498db','#e74c3c','#2ecc71'],
            edgecolor='white', width=0.5)
axes[1].set_title('Avg Selling Price by Fuel Type',
                  fontsize=13, fontweight='bold')
axes[1].set_ylabel('Avg Price (Lakhs)', fontsize=11)
for i, val in enumerate(fuel_price.values):
    axes[1].text(i, val + 0.2, f'{val:.2f}L',
                 ha='center', fontsize=11, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Chart 4 — Fuel Type Analysis',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('chart4_fuel_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ── STEP 6: Sample Prediction ────────────────────────────────────
print("\n--- Sample Car Price Prediction ---")
rf_model = models['Random Forest']
sample = pd.DataFrame([{
    'Present_Price'   : 5.59,
    'Car_Age'         : 6,
    'Driven_kms'      : 27000,
    'Fuel_Type_enc'   : 1,
    'Selling_type_enc': 0,
    'Transmission_enc': 1,
    'Owner'           : 0
}])
predicted_price = rf_model.predict(sample)[0]
print(f"  Present Price : 5.59 Lakhs")
print(f"  Car Age       : 6 years")
print(f"  Driven KMs    : 27,000 km")
print(f"  Fuel Type     : Petrol")
print(f"  Transmission  : Manual")
print(f"  Predicted Selling Price: {predicted_price:.2f} Lakhs")

print("\n" + "=" * 55)
print("  KEY INSIGHTS")
print("=" * 55)
print(f"\n1. Best model     : {best}")
print(f"2. R² Score       : {results[best]['r2']*100:.2f}%")
print(f"3. MAE            : {results[best]['mae']:.2f} Lakhs")
print(f"4. Avg car price  : {df['Selling_Price'].mean():.2f} Lakhs")
print(f"5. Most cars use  : {df['Fuel_Type'].value_counts().idxmax()} fuel")
print(f"\nAll 4 charts saved in your project folder!")