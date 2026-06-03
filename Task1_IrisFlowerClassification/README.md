# 🌸 Iris Flower Classification

A Machine Learning project that classifies Iris flowers into three species —
**Setosa**, **Versicolor**, and **Virginica** — based on their petal and sepal measurements.

Built as part of the **CodeAlpha Data Science Internship**.

---

## 📌 Project Overview

The Iris dataset contains 150 flower samples with 4 measurements each:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

A **Support Vector Machine (SVM)** classifier is trained to predict the species
of a flower based on these measurements.

---

## 📁 Project Structure

```
IRIS_PROJECT/
│
├── iris_classifier.py        # Main ML model — training and evaluation
├── iris_charts.py            # Visualizations — 4 charts shown one by one
├── Iris.csv                  # Dataset downloaded from Kaggle
│
├── chart1_confusion_matrix.png   # Confusion matrix chart
├── chart2_petal_scatter.png      # Petal length vs petal width scatter plot
├── chart3_sepal_scatter.png      # Sepal length vs sepal width scatter plot
└── chart4_performance.png        # Model performance metrics bar chart
```

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Prime7777/CodeAlpha_IrisFlowerClassification.git
cd CodeAlpha_IrisFlowerClassification
```

### 2. Install required libraries
```bash
pip install scikit-learn pandas numpy matplotlib
```

### 3. Run the classifier
```bash
python iris_classifier.py
```

### 4. Run the charts
```bash
python iris_charts.py
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 94.74% |
| **Precision** | 94.87% |
| **Recall** | 94.87% |
| **F1 Score** | 94.87% |

### Per-class Performance

| Species | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Iris-setosa | 1.00 | 1.00 | 1.00 |
| Iris-versicolor | 0.92 | 0.92 | 0.92 |
| Iris-virginica | 0.92 | 0.92 | 0.92 |

---

## 📈 Visualizations

### Confusion Matrix
Shows how many flowers were correctly and incorrectly classified.
- Setosa: **perfectly classified** (12/12)
- Versicolor: 12 correct, 1 misclassified
- Virginica: 12 correct, 1 misclassified

### Scatter Plots
- **Petal chart:** Setosa is clearly separated from the other two species
- **Sepal chart:** More overlap between Versicolor and Virginica

---

## 🛠️ Libraries Used

| Library | Purpose |
|---------|---------|
| `scikit-learn` | ML model, train/test split, evaluation metrics |
| `pandas` | Loading and processing the CSV dataset |
| `numpy` | Numerical operations |
| `matplotlib` | Data visualizations and charts |

---

## 🧠 Concepts Learned

- **Train/Test Split** — 75% training, 25% testing (stratified)
- **Feature Scaling** — StandardScaler for normalizing measurements
- **SVM Classifier** — Support Vector Machine with RBF kernel
- **Model Evaluation** — Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix** — Visual representation of predictions vs actual

---

## 📂 Dataset

- **Source:** [Kaggle — Iris Flower Dataset](https://www.kaggle.com/datasets/uciml/iris)
- **Samples:** 150 flowers (50 per species)
- **Features:** 4 numerical measurements
- **Classes:** 3 species (Setosa, Versicolor, Virginica)

---

## 👤 Author

**Primu**
CodeAlpha Data Science Internship
