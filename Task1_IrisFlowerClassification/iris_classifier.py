# ── IRIS FLOWER CLASSIFIER (Kaggle Dataset) ─────────────────────
# Loading from your downloaded Iris.csv file

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── STEP 1: Load YOUR Kaggle CSV file ───────────────────────────
df = pd.read_csv("Iris.csv")

print("Dataset loaded from Iris.csv!")
print(f"Total flowers: {len(df)}")
print(f"\nFirst 5 rows of your dataset:")
print(df.head())
print(f"\nColumn names: {list(df.columns)}")
print()

# ── STEP 2: Prepare features and labels ─────────────────────────
# Drop the 'Id' column (not useful for prediction)
df = df.drop("Id", axis=1)

# X = the 4 measurements, y = the species name
X = df.drop("Species", axis=1).values
y = df["Species"].values

print(f"Species found: {list(set(y))}")
print()

# ── STEP 3: Split into training and test sets ───────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
print()

# ── STEP 4: Scale the features ──────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Features scaled.")
print()

# ── STEP 5: Train the model ─────────────────────────────────────
model = SVC(kernel="rbf", random_state=42)
model.fit(X_train, y_train)

print("Model trained!")
print()

# ── STEP 6: Evaluate the model ──────────────────────────────────
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print()

print("Detailed Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print("(Rows = Actual species, Columns = Predicted species)")
print(confusion_matrix(y_test, y_pred))
print()

# ── STEP 7: Predict a new flower ────────────────────────────────
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)[0]
print(f"New flower measurements: sepal=5.1x3.5cm, petal=1.4x0.2cm")
print(f"Predicted species: {prediction}")
