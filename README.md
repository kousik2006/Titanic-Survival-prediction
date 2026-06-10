<div align="center">

# 🚢 Titanic Survival Prediction
### Manual Preprocessing — Without sklearn Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

>
> Building a Titanic survival classifier **entirely without** sklearn.pipeline.Pipeline —
> every preprocessing step is coded manually to deeply understand what happens inside a pipeline.

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Preprocessing Pipeline (Manual)](#-preprocessing-pipeline-manual)
- [Feature Matrix](#-feature-matrix)
- [Models](#-models)
- [Results & Evaluation](#-results--evaluation)
- [Bugs Found & Fixed](#-bugs-found--fixed)
- [Key Learnings](#-key-learnings)
- [Manual vs Pipeline — Comparison](#-manual-vs-pipeline--comparison)
- [How to Run](#-how-to-run)
- [Dependencies](#-dependencies)
- [Repository Structure](#-repository-structure)
- [Next Steps](#-next-steps)

---

## 📌 Project Overview

The Titanic dataset is one of the most well-known beginner ML datasets. The goal is to predict whether a passenger **survived** the Titanic disaster based on features like passenger class, sex, age, fare, and port of embarkation.

This notebook is **not** about achieving the highest accuracy. Its purpose is to:

1. **Understand** every preprocessing step explicitly — what happens to each column and why.
2. **Learn** the correct way to apply `SimpleImputer`, `OneHotEncoder`, and `train_test_split` without leaking test data into training.
3. **Manually assemble** the final feature matrix using `np.concatenate`, which teaches the exact column-ordering discipline that `ColumnTransformer` automates.
4. **Compare** two classifiers — Decision Tree and Gradient Boosting — and observe the effect of model complexity on overfitting.
5. **Motivate** the need for `sklearn.pipeline.Pipeline` by experiencing the manual approach's verbosity and error-proneness first-hand.

This notebook is **Part 1** of a two-part series. Part 2 (Day 30) will replicate this exact workflow using a proper sklearn Pipeline.

---

## 📂 Dataset

**Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

```
train.csv — 891 rows × 12 columns
```

### Column Descriptions

| Column | Type | Description |
|---|---|---|
| `PassengerId` | int | Unique passenger identifier |
| `Survived` | int (0/1) | **Target** — 0 = Did not survive, 1 = Survived |
| `Pclass` | int (1/2/3) | Ticket class — proxy for socio-economic status |
| `Name` | string | Passenger full name |
| `Sex` | string | male / female |
| `Age` | float | Age in years — **has missing values (~19.8%)** |
| `SibSp` | int | Number of siblings/spouses aboard |
| `Parch` | int | Number of parents/children aboard |
| `Ticket` | string | Ticket number |
| `Fare` | float | Passenger fare |
| `Cabin` | string | Cabin number — **heavily missing (~77.1%)** |
| `Embarked` | string | Port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton — **2 missing** |

### Missing Values Summary

| Column | Missing Count | Missing % | Action Taken |
|---|---|---|---|
| `Age` | ~177 | ~19.8% | Mean imputation |
| `Cabin` | ~687 | ~77.1% | Column dropped |
| `Embarked` | 2 | ~0.2% | Mode imputation |

---

## 🔍 Exploratory Data Analysis

### 1. Basic Inspection
```python
df.sample(5)          # Random sample of 5 rows
df.info()             # Dtypes, non-null counts
df.describe()         # Statistical summary for numeric columns
df.isnull().sum()     # Missing value counts per column
df.duplicated().sum() # Duplicate row check
```

### 2. Embarked Distribution
```python
df['Embarked'].value_counts()
# S    644   (Southampton — most common)
# C    168   (Cherbourg)
# Q     77   (Queenstown)
```

### 3. Correlation Analysis
```python
df.corr(numeric_only=True)
```

Notable correlations with `Survived`:

| Feature | Correlation | Interpretation |
|---|---|---|
| `Pclass` | Negative | Lower class = lower survival rate |
| `Fare` | Positive | Higher fare = higher survival rate |
| `Sex` | Strong (after encoding) | Strongest single predictor |
| `SibSp` / `Parch` | Weak | Minor influence |

### 4. Survival by Sex
```python
sns.heatmap(pd.crosstab(df['Sex'], df['Survived']))
```

| Sex | Did Not Survive | Survived |
|---|---|---|
| Female | 81 | 233 |
| Male | 468 | 109 |

Females had a significantly higher survival rate — the strongest signal in the dataset ("women and children first").

### 5. Pairplot
```python
sns.pairplot(df, hue="Survived")
```
Visualises pairwise relationships between all numeric features, coloured by survival status. Useful for spotting class separability.

---

## ⚙️ Preprocessing Pipeline (Manual)

All steps are done manually without `ColumnTransformer` or `Pipeline`.

### Step 1 — Drop Irrelevant Columns

```python
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
```

| Column | Reason for Dropping |
|---|---|
| `PassengerId` | Unique row index — zero predictive signal |
| `Name` | High cardinality text; title extraction would need feature engineering |
| `Ticket` | Mixed alphanumeric format with no consistent pattern |
| `Cabin` | 77% missing — too sparse to impute reliably |

### Step 2 — Train-Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # Preserves class ratio in both splits
)
# Train: 712 rows | Test: 179 rows
```

The split is done **before any preprocessing** to prevent data leakage. The test set must remain completely unseen during imputer/encoder fitting.

### Step 3 — Imputation

```python
from sklearn.impute import SimpleImputer

si_age      = SimpleImputer()                          # strategy='mean' by default
si_embarked = SimpleImputer(strategy='most_frequent')  # fills with 'S'
```

| Column | Strategy | Fit On | Transform |
|---|---|---|---|
| `Age` | Mean | x_train only | Both x_train and x_test |
| `Embarked` | Most frequent | x_train only | Both x_train and x_test |

```python
# Fit on train, transform both
x_train_age      = si_age.fit_transform(x_train[['Age']])
x_train_embarked = si_embarked.fit_transform(x_train[['Embarked']])

x_test_age      = si_age.transform(x_test[['Age']])            # .transform() only!
x_test_embarked = si_embarked.transform(x_test[['Embarked']]) # .transform() only!
```

### Step 4 — One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

ohe_sex      = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_embarked = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
```

| Column | Output Columns | Output Shape |
|---|---|---|
| `Sex` | female, male | (n, 2) |
| `Embarked` | C, Q, S | (n, 3) |

```python
x_train_sex      = ohe_sex.fit_transform(x_train[['Sex']])
x_train_embarked = ohe_embarked.fit_transform(x_train_embarked)

x_test_sex      = ohe_sex.transform(x_test[['Sex']])
x_test_embarked = ohe_embarked.transform(x_test_embarked)
```

### Step 5 — Remove Original Categorical Columns

```python
x_train_rem = x_train.drop(columns=['Sex', 'Age', 'Embarked'])
x_test_rem  = x_test.drop(columns=['Sex', 'Age', 'Embarked'])
# Remaining numeric: Pclass, SibSp, Parch, Fare
```

### Step 6 — Assemble Final Feature Matrix

```python
x_train_transformed = np.concatenate(
    (x_train_rem, x_train_age, x_train_embarked, x_train_sex), axis=1
)
x_test_transformed = np.concatenate(
    (x_test_rem, x_test_age, x_test_embarked, x_test_sex), axis=1  # identical order!
)
```

> ⚠️ The column order in `np.concatenate` must be **identical** for both train and test.
> Swapping two arrays here silently passes wrong values to every prediction — no exception is raised.

---

## 📐 Feature Matrix

After all preprocessing, the final input matrix has **10 columns**:

| Index | Feature | Source |
|---|---|---|
| 0 | `Pclass` | Original numeric |
| 1 | `SibSp` | Original numeric |
| 2 | `Parch` | Original numeric |
| 3 | `Fare` | Original numeric |
| 4 | `Age` (imputed) | Mean-imputed |
| 5 | `Embarked_C` | One-hot encoded |
| 6 | `Embarked_Q` | One-hot encoded |
| 7 | `Embarked_S` | One-hot encoded |
| 8 | `Sex_female` | One-hot encoded |
| 9 | `Sex_male` | One-hot encoded |

**Shape:** `x_train_transformed` → (712, 10) | `x_test_transformed` → (179, 10)

---

## 🤖 Models

### 1. Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(x_train_transformed, y_train)
```

**How it works:** Recursively splits the feature space using the threshold that best separates classes (Gini impurity). Without depth constraints, it grows until every training sample is perfectly classified — memorising noise and outliers.

**Known limitation:** No `max_depth`, `min_samples_split`, or `min_samples_leaf` is set here, which causes severe overfitting.

| Metric | Train Score | Test Score |
|---|---|---|
| Accuracy | ~100% | ~75–78% |

### 2. Gradient Boosting Classifier

```python
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(
    n_estimators=100,    # 100 weak learners (shallow trees)
    learning_rate=0.1,   # Shrinkage — each tree contributes 10% of its correction
    random_state=42
)
```

**How it works:** Builds trees **sequentially**. Each new tree corrects the residual errors of the current ensemble. The `learning_rate` acts as regularisation — smaller values force the model to learn slowly, reducing overfitting at the cost of needing more trees.

**Advantage:** Better generalisation than a single Decision Tree due to the sequential error-correction mechanism.

| Metric | Train Score | Test Score |
|---|---|---|
| Accuracy | ~88–90% | ~82–84% |

---

## 📊 Results & Evaluation

Both models are evaluated with:

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Overfitting check
print("Train:", clf.score(x_train_transformed, y_train))
print("Test :", clf.score(x_test_transformed, y_test))
```

### Summary Comparison

| Model | Train Acc | Test Acc | Gap | Verdict |
|---|---|---|---|---|
| Decision Tree (unconstrained) | ~100% | ~75–78% | ~22–25% | Heavily overfit |
| Gradient Boosting | ~88–90% | ~82–84% | ~5–6% | Acceptable generalisation |

> Exact values will vary slightly. Run the notebook to see the actual outputs.

---

## 🐛 Bugs Found & Fixed

Three significant bugs were identified and corrected during development:

### Bug 1 — `fit_transform` on Test Data (Data Leakage)

```python
# WRONG — refits the imputer using test set statistics
x_test_age = si_age.fit_transform(x_test[['Age']])

# CORRECT — uses the mean learned from training data only
x_test_age = si_age.transform(x_test[['Age']])
```

**Root cause:** `fit_transform` = `fit` + `transform`. Calling it on the test set computes a new mean from test data, overriding what was learned from training.

**Impact:** Subtle data leakage. The imputer uses a different fill value for test data than training data, which is statistically incorrect and makes test evaluation unreliable.

---

### Bug 2 — Column Order Mismatch in `np.concatenate` (Most Critical)

```python
# WRONG — embarked and sex are SWAPPED for test
x_train_transformed = np.concatenate((x_train_rem, x_train_age, x_train_embarked, x_train_sex), axis=1)
x_test_transformed  = np.concatenate((x_test_rem,  x_test_age,  x_test_sex, x_test_embarked), axis=1)
#                                                                ^^^^^^^^^^^ ^^^^^^^^^^^^^^^  swapped!

# CORRECT — identical order for both
x_test_transformed = np.concatenate((x_test_rem, x_test_age, x_test_embarked, x_test_sex), axis=1)
```

**Root cause:** A simple copy-paste swap of two variable names. NumPy arrays have no column names — they are just numbers. The model is told nothing about the swap.

**Impact:** The most damaging bug. The model reads `Sex` values where it expects `Embarked` and vice versa during every test prediction. No error is thrown. The model silently produces wrong predictions, causing a large unexplained accuracy drop.

---

### Bug 3 — Unconstrained Decision Tree (Overfitting)

```python
# PROBLEMATIC — no regularisation, memorises training data
clf = DecisionTreeClassifier()

# BETTER — add constraints to prevent overfitting
clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, min_samples_split=10)
```

**Root cause:** Default `DecisionTreeClassifier` has `max_depth=None`, meaning it grows until all leaves are pure or contain fewer than `min_samples_split` samples (default: 2).

**Impact:** Train accuracy ~100%, test accuracy ~20% lower. The model has memorised the training set and cannot generalise to new data.

---

## 💡 Key Learnings

| # | Concept | Lesson |
|---|---|---|
| 1 | **Fit/Transform discipline** | `fit_transform()` on train only. `transform()` on everything else. Fitting on test data leaks distribution statistics and is conceptually wrong. |
| 2 | **Column order is contract** | `np.concatenate` stacks raw arrays — it has no column names. If order differs between train and test, predictions are silently corrupted with zero exceptions raised. |
| 3 | **Overfitting vs generalisation** | Unconstrained Decision Tree memorises training data (100% train acc). Gradient Boosting corrects incrementally and generalises better. |
| 4 | **Stratified splits** | `stratify=y` ensures both splits have the same ratio of survivors, preventing a skewed or unrepresentative test set. |
| 5 | **Why Pipelines exist** | Bugs 1 and 2 above are structurally impossible in a `Pipeline` + `ColumnTransformer` setup. The pipeline enforces correct fit/transform flow and manages column ordering automatically. |

---

## ⚖️ Manual vs Pipeline — Comparison

| Aspect | Manual (This Notebook) | With `Pipeline` + `ColumnTransformer` |
|---|---|---|
| Lines of code | ~30 lines for preprocessing | ~10 lines |
| Data leakage risk | High — easy to `fit_transform` test data | None — pipeline enforces correct flow |
| Column order bugs | High — `np.concatenate` ordering is manual | None — `ColumnTransformer` handles it |
| Readability | Verbose, many intermediate variables | Concise, single pipeline object |
| `cross_val_score` compatible | No — requires re-running manually | Yes — plug directly into CV |
| `GridSearchCV` compatible | No | Yes |
| Learning value | Very high — every step is visible | Lower — internal details abstracted |

> **Takeaway:** Do the manual approach once to understand the internals. Use Pipelines for everything real.

---

## 🚀 How to Run

### Option A — Google Colab

1. Upload `Titanic_without_pipeline.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload `train.csv` to your Google Drive
3. Update the dataset path in Cell 2:
   ```python
   df = pd.read_csv("/content/drive/MyDrive/your_path/train.csv")
   ```
4. Runtime → Run All

### Option B — Local Jupyter

```bash
# 1. Clone the repo
git clone https://github.com/your-username/titanic-without-pipeline.git
cd titanic-without-pipeline

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook Titanic_without_pipeline.ipynb
```

5. Update the CSV path in Cell 2 to your local path:
   ```python
   df = pd.read_csv("train.csv")
   ```

---

## 🛠️ Dependencies

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
seaborn>=0.11
matplotlib>=3.4
```

Install all at once:
```bash
pip install numpy pandas scikit-learn seaborn matplotlib
```

| Library | Purpose |
|---|---|
| `pandas` | Data loading, EDA, DataFrame manipulation |
| `numpy` | Array operations, `np.concatenate` for feature assembly |
| `scikit-learn` | `SimpleImputer`, `OneHotEncoder`, `train_test_split`, classifiers, metrics |
| `seaborn` | Pairplot and heatmap visualisations |
| `matplotlib` | Underlying plotting engine |

---

## 📁 Repository Structure

```
titanic-without-pipeline/
│
├── Titanic_without_pipeline.ipynb   ← Main notebook
├── train.csv                        ← Dataset (download from Kaggle)
├── requirements.txt                 ← Python dependencies
└── README.md                        ← This file
```

---

## 🔗 Next Steps

- [ ] **Day 30** — Replicate this workflow using `sklearn.pipeline.Pipeline` + `ColumnTransformer`
- [ ] Add `GridSearchCV` for Decision Tree hyperparameter tuning (`max_depth`, `min_samples_leaf`)
- [ ] Add `RandomForestClassifier` for comparison
- [ ] Feature engineering ideas:
  - Extract passenger title from `Name` (Mr, Mrs, Miss, Master)
  - Create `FamilySize = SibSp + Parch + 1`
  - Create `IsAlone` binary flag
- [ ] Apply `StandardScaler` to numeric features and observe effect on tree-based models
- [ ] Use `cross_val_score` for a more reliable accuracy estimate

---

## 🙌 Acknowledgements

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) for the dataset
- [scikit-learn Documentation](https://scikit-learn.org/stable/) — `SimpleImputer`, `OneHotEncoder`, classifiers reference
- **100 Days of ML** — the broader learning series this notebook belongs to

---

<div align="center">

⭐ If this helped you understand manual preprocessing, consider giving it a star!

</div>
