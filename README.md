# 🚢 Titanic Survival Prediction — Manual Preprocessing (Without Pipeline)

> **Day 29 of 100 Days of ML**
> Exploring manual feature preprocessing step-by-step, before moving to `sklearn` Pipelines.

---

## 📌 Project Overview

This notebook builds a binary classification model to predict passenger survival on the Titanic. The focus is on performing **every preprocessing step manually** — imputation, encoding, and feature assembly — without using `sklearn.pipeline.Pipeline` or `ColumnTransformer`. This makes the internals explicit and helps understand what a pipeline does under the hood.

Two classifiers are trained and compared:
- Decision Tree Classifier
- Gradient Boosting Classifier

---

## 📂 Dataset

**Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

| Property | Value |
|---|---|
| File | `train.csv` |
| Rows | 891 |
| Target Column | `Survived` (0 = No, 1 = Yes) |

---

## 🗂️ Project Workflow

```
Raw Data
   │
   ├── EDA (pairplot, heatmap, correlation, null check)
   │
   ├── Feature Selection (drop irrelevant columns)
   │
   ├── Train-Test Split (80/20, stratified)
   │
   ├── Imputation
   │     ├── Age  → mean (fit on train, transform both)
   │     └── Embarked → most_frequent (fit on train, transform both)
   │
   ├── One-Hot Encoding
   │     ├── Sex
   │     └── Embarked
   │
   ├── Feature Assembly (np.concatenate)
   │
   └── Model Training & Evaluation
         ├── Decision Tree Classifier
         └── Gradient Boosting Classifier
```

---

## 🔧 Preprocessing Steps

### Dropped Columns
The following columns were removed before modeling as they are identifiers or have too many missing values to be useful:

| Column | Reason |
|---|---|
| `PassengerId` | Unique identifier, no signal |
| `Name` | High cardinality, not used |
| `Ticket` | High cardinality, not used |
| `Cabin` | ~77% missing values |

### Features Used for Training

| Feature | Type | Treatment |
|---|---|---|
| `Pclass` | Numeric | Used as-is |
| `Sex` | Categorical | One-Hot Encoded |
| `Age` | Numeric | Mean imputation → used as-is |
| `SibSp` | Numeric | Used as-is |
| `Parch` | Numeric | Used as-is |
| `Fare` | Numeric | Used as-is |
| `Embarked` | Categorical | Mode imputation → One-Hot Encoded |

### Key Preprocessing Rules Followed
- Imputers are **fit only on training data** (`fit_transform` on train, `.transform()` on test) to prevent data leakage.
- Column order in `np.concatenate` is kept **identical** for both train and test arrays.

---

## 🤖 Models

### 1. Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
```
- Unconstrained tree (no `max_depth`) — prone to overfitting.
- Train score is significantly higher than test score.

### 2. Gradient Boosting Classifier
```python
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
```
- Ensemble method, more robust to overfitting.
- Better generalisation than a single Decision Tree.

---

## 📊 Evaluation Metrics

Both models are evaluated using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix**
- **Train vs Test score comparison** (to detect overfitting)

---

## 💡 Key Learnings

1. **`fit_transform` vs `transform`** — Calling `fit_transform` on the test set leaks information from the test distribution into your preprocessing. Always fit only on training data.

2. **Column order consistency** — When manually assembling transformed features with `np.concatenate`, the column order must be **exactly the same** for train and test. A swap silently corrupts all predictions.

3. **Decision Trees overfit** — Without constraining `max_depth` or `min_samples_leaf`, a Decision Tree memorises the training data (train ≈ 100%, test much lower). Gradient Boosting handles this more gracefully.

4. **Why Pipelines exist** — Doing all of this manually is verbose and error-prone. These exact bugs (`fit_transform` on test, column order mismatch) are structurally impossible when using `sklearn.pipeline.Pipeline` + `ColumnTransformer`.

---

## 🚀 How to Run

1. Clone the repo and open the notebook in Jupyter or Google Colab.

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn seaborn matplotlib
   ```

3. Update the dataset path in the second cell:
   ```python
   df = pd.read_csv("path/to/train.csv")
   ```

4. Run all cells top to bottom.

---

## 🛠️ Dependencies

| Library | Version |
|---|---|
| Python | ≥ 3.8 |
| pandas | ≥ 1.3 |
| numpy | ≥ 1.21 |
| scikit-learn | ≥ 1.0 |
| seaborn | ≥ 0.11 |
| matplotlib | ≥ 3.4 |

---

## 📁 Repository Structure

```
├── Titanic_without_pipeline.ipynb   # Main notebook
├── train.csv                        # Dataset (download from Kaggle)
└── README.md
```

---

## 🔗 Next Steps

- [ ] Refactor the same workflow using `sklearn.pipeline.Pipeline`
- [ ] Add hyperparameter tuning with `GridSearchCV`
- [ ] Try `RandomForestClassifier` and `XGBoost`
- [ ] Add feature engineering (e.g., family size, title extraction from Name)

---

## 📜 License

This project is part of a personal learning series and is open for reference and learning purposes.h
