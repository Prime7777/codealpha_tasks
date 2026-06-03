# ── IRIS CHARTS - One by One ─────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# ── Load and prepare data ────────────────────────────────────────
df = pd.read_csv("Iris.csv")
df = df.drop("Id", axis=1)
X = df.drop("Species", axis=1).values
y = df["Species"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="rbf", random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
colors = ["#2ecc71", "#3498db", "#e74c3c"]

# ── CHART 1: Confusion Matrix ────────────────────────────────────
print("Showing Chart 1 of 4 - Confusion Matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=species)
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["Setosa", "Versicolor", "Virginica"], fontsize=13)
ax.set_yticklabels(["Setosa", "Versicolor", "Virginica"], fontsize=13)
ax.set_xlabel("Predicted", fontsize=14, labelpad=15)
ax.set_ylabel("Actual", fontsize=14, labelpad=15)
ax.set_title("Chart 1 — Confusion Matrix", fontsize=16, pad=20)
for i in range(3):
    for j in range(3):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=28,
            fontweight="bold",
            color="white" if cm[i, j] > 5 else "black",
        )
plt.tight_layout()
plt.savefig("chart1_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# ── CHART 2: Petal Length vs Petal Width ─────────────────────────
print("Showing Chart 2 of 4 - Petal Scatter Plot...")
fig, ax = plt.subplots(figsize=(9, 6))
for sp, col in zip(species, colors):
    mask = df["Species"] == sp
    ax.scatter(
        df[mask]["PetalLengthCm"],
        df[mask]["PetalWidthCm"],
        label=sp.replace("Iris-", ""),
        color=col,
        alpha=0.8,
        s=90,
    )
ax.set_xlabel("Petal Length (cm)", fontsize=14, labelpad=10)
ax.set_ylabel("Petal Width (cm)", fontsize=14, labelpad=10)
ax.set_title("Chart 2 — Petal Length vs Petal Width", fontsize=16, pad=20)
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig("chart2_petal_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# ── CHART 3: Sepal Length vs Sepal Width ─────────────────────────
print("Showing Chart 3 of 4 - Sepal Scatter Plot...")
fig, ax = plt.subplots(figsize=(9, 6))
for sp, col in zip(species, colors):
    mask = df["Species"] == sp
    ax.scatter(
        df[mask]["SepalLengthCm"],
        df[mask]["SepalWidthCm"],
        label=sp.replace("Iris-", ""),
        color=col,
        alpha=0.8,
        s=90,
    )
ax.set_xlabel("Sepal Length (cm)", fontsize=14, labelpad=10)
ax.set_ylabel("Sepal Width (cm)", fontsize=14, labelpad=10)
ax.set_title("Chart 3 — Sepal Length vs Sepal Width", fontsize=16, pad=20)
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig("chart3_sepal_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# ── CHART 4: Model Performance Metrics ───────────────────────────
print("Showing Chart 4 of 4 - Model Performance...")
fig, ax = plt.subplots(figsize=(9, 6))
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average="macro"),
    "Recall": recall_score(y_test, y_pred, average="macro"),
    "F1 Score": f1_score(y_test, y_pred, average="macro"),
}
bars = ax.bar(
    metrics.keys(),
    metrics.values(),
    color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"],
    edgecolor="white",
    linewidth=1.5,
    width=0.5,
)
ax.set_ylim(0, 1.15)
ax.set_title("Chart 4 — Model Performance Metrics", fontsize=16, pad=20)
ax.set_ylabel("Score", fontsize=14, labelpad=10)
ax.tick_params(axis="x", labelsize=13)
for bar, val in zip(bars, metrics.values()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.03,
        f"{val:.2%}",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
plt.tight_layout()
plt.savefig("chart4_performance.png", dpi=150, bbox_inches="tight")
plt.show()

print()
print("All 4 charts saved in your iris_project folder!")
