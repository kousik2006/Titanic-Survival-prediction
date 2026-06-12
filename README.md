# 🚢 Titanic Survival Prediction — Manual Preprocessing vs Sklearn Pipeline

> A hands-on comparison of doing preprocessing manually step-by-step versus
> wrapping the exact same logic inside a clean, reproducible `sklearn` Pipeline.

---

## 📌 Project Overview

This project tackles the classic **Titanic survival prediction** problem, but the
real learning goal here isn't just the model — it's understanding *how* to
structure an ML workflow properly.

Two separate notebooks are maintained side-by-side:

| Notebook | Approach |
|----------|----------|
| `without_pipeline.ipynb` | Manual preprocessing — impute, encode, scale, and concatenate by hand |
| `Titanic_by_pipelining.ipynb` | Same preprocessing logic, refactored into a clean `sklearn` Pipeline |

Both notebooks use the same dataset, the same features, the same transformations,
and the same model. The only difference is *how* the preprocessing is organized.
This makes it easy to see exactly what a Pipeline buys you.

---

## 📂 Dataset

**Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

The training file (`train.csv`) contains **891 rows** and **12 columns**.

### Raw Features

| Column | Type | Description |
|--------|------|-------------|
| `PassengerId` | int | Unique passenger identifier |
| `Survived` | int (0/1) | Target — 0 = did not survive, 1 = survived |
| `Pclass` | int (1/2/3) | Ticket class — proxy for socio-economic status |
| `Name` | string | Passenger name |
| `Sex` | string | `male` / `female` |
| `Age` | float | Age in years (has missing values) |
| `SibSp` | int | No. of siblings/spouses aboard |
| `Parch` | int | No. of parents/children aboard |
| `Ticket` | string | Ticket number |
| `Fare` | float | Passenger fare |
| `Cabin` | string | Cabin number (heavily missing) |
| `Embarked` | string | Port of embarkation — C = Cherbourg, Q = Queenstown, S = Southampton |

### Missing Values (in raw data)

| Column | Missing Count | Missing % |
|--------|--------------|-----------|
| `Cabin` | 687 | 77.1% |
| `Age` | 177 | 19.9% |
| `Embarked` | 2 | 0.2% |

### Columns Dropped

`PassengerId`, `Name`, `Ticket`, and `Cabin` are dropped before any modeling:

- `PassengerId` — arbitrary index, no predictive signal
- `Name` — raw text; title extraction could be done but is out of scope here
- `Ticket` — alphanumeric with no consistent structure
- `Cabin` — 77% missing; imputing this much data would introduce more noise than signal

After dropping, the feature set is:
**`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`**

---

## 🔍 Exploratory Data Analysis

A few key observations from the EDA in `without_pipeline.ipynb`:

### Survival by Sex
```
Survived    0    1
Sex
female     81  233   →  74.2% survival rate
male      468  109   →  18.9% survival rate
```
Sex is one of the strongest predictors. The "women and children first" protocol
is clearly visible in the data.

### Class Imbalance
```
Survived = 0  →  549 passengers  (61.6%)
Survived = 1  →  342 passengers  (38.4%)
```
The dataset is moderately imbalanced. `stratify=y` is used in `train_test_split`
to preserve this ratio in both train and test sets.

### Age Distribution
Age has **177 missing values (19.9%)**. These are imputed using the **mean age**
(~29.8 years). Mean imputation is a reasonable baseline here, though more
sophisticated approaches (e.g., median, or model-based imputation grouped by
`Pclass`) could be explored.

### Embarked Distribution
```
S (Southampton)  →  644 passengers  (72.3%)
C (Cherbourg)    →  168 passengers  (18.9%)
Q (Queenstown)   →   77 passengers  (8.6%)
```
Only 2 missing values — imputed with the mode (`S`).

---

## 🛠️ Preprocessing Steps

Both notebooks apply the exact same preprocessing, just structured differently.

### Step 1 — Imputation
- `Age`: filled with **mean** using `SimpleImputer(strategy='mean')`
- `Embarked`: filled with **mode** using `SimpleImputer(strategy='most_frequent')`

### Step 2 — One-Hot Encoding
- `Sex`: encoded into 2 binary columns (`female`, `male`)
- `Embarked`: encoded into 3 binary columns (`C`, `Q`, `S`)

`handle_unknown='ignore'` is set so unseen categories at prediction time don't
cause errors. `sparse_output=False` returns a dense numpy array.

### Step 3 — Feature Scaling
All features are scaled to the **[0, 1]** range using `MinMaxScaler`.
This is required before `SelectKBest(chi2)` because chi-squared scores
are not defined for negative values.

### Step 4 — Feature Selection
`SelectKBest(score_func=chi2, k=8)` keeps the **8 most statistically
significant features** out of the 10 available after encoding.
Chi-squared tests measure the dependency between each feature and the target,
discarding features with low signal.

### Step 5 — Classification
`DecisionTreeClassifier` is used as the final estimator.

---

## ⚙️ Approach 1 — Manual Preprocessing (`without_pipeline.ipynb`)

In this notebook, every preprocessing step is done by hand:

```python
# Impute
si_age      = SimpleImputer()
si_embarked = SimpleImputer(strategy="most_frequent")

x_train_age      = si_age.fit_transform(x_train[['Age']])
x_train_embarked = si_embarked.fit_transform(x_train[['Embarked']])

# Encode
ohe_sex      = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_embarked = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

x_train_sex      = ohe_sex.fit_transform(x_train[['Sex']])
x_train_embarked = ohe_embarked.fit_transform(x_train_embarked)

# Drop original columns and concatenate everything manually
x_train_rem = x_train.drop(columns=["Sex", "Age", "Embarked"])
x_train_transformed = np.concatenate(
    (x_train_rem, x_train_age, x_train_embarked, x_train_sex), axis=1
)
```

This works, but it has several real problems:

**① Code duplication.** Every transformation has to be written twice — once
for `x_train` and once for `x_test`. With 4+ steps, that's a lot of repeated
code that can easily get out of sync.

**② Risk of data leakage.** Each imputer and encoder must be `fit` on training
data only, and then only `transform`-ed on test data. It's easy to accidentally
call `fit_transform` on test data instead of just `transform`, which leaks
test distribution into the training process. (This actually happened in Cell 38
of this notebook — `si_age.fit_transform(x_test[['Age']])` was used instead of
`si_age.transform(x_test[['Age']])`.)

**③ Manual column tracking.** After encoding, the original `Sex`, `Age`, and
`Embarked` columns must be manually dropped and the new arrays concatenated in
the right order. Getting the order wrong causes silent bugs.

**④ Not deployable as-is.** To make predictions on new data, you would need to
manually re-apply every step in the exact right order with the fitted objects.
There is no single object you can call `.predict()` on.

### Results (Manual Approach)

| Model | Train Accuracy | Test Accuracy |
|-------|---------------|---------------|
| `DecisionTreeClassifier` | **98.3%** | **60.9%** |
| `GradientBoostingClassifier` | — | **60.9%** |

The Decision Tree is **severely overfitting** — 98.3% on training data vs 60.9%
on test data is a classic sign that the tree is memorizing the training set.
This is expected without any depth limit or regularization on the tree.

```
              precision    recall  f1-score   support
           0       0.66      0.75      0.70       110
           1       0.49      0.39      0.44        69
    accuracy                           0.61       179
```

---

## ⚙️ Approach 2 — sklearn Pipeline (`Titanic_by_pipelining.ipynb`)

The same preprocessing is refactored into a `sklearn` Pipeline with 5 named steps:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("trf1", trf1),   # Imputation
    ("trf2", trf2),   # One-Hot Encoding
    ("trf3", trf3),   # MinMax Scaling
    ("trf4", trf4),   # Feature Selection (SelectKBest chi2)
    ("trf5", trf5),   # DecisionTreeClassifier
])

pipe.fit(x_train, y_train)    # All 5 steps, in one call
pipe.predict(x_test)          # Applies all transforms + predicts, in one call
```

Or equivalently using `make_pipeline` (no need to name each step manually):

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(trf1, trf2, trf3, trf4, trf5)
```

### Why Pipeline is better

**① No leakage by design.** When you call `pipe.fit(x_train, y_train)`, sklearn
calls `fit_transform` on each step using only the training data. When you call
`pipe.predict(x_test)`, it calls `transform` (not `fit_transform`) for every
intermediate step automatically. You can't accidentally leak.

**② Single deployable object.** The fitted `pipe` object contains every fitted
transformer and the final model. Save it with `joblib.dump(pipe, 'model.pkl')`,
load it anywhere, and call `pipe.predict(new_data)`. Nothing else needed.

**③ Clean, readable code.** 5 lines to define the full ML workflow instead of
30+ lines of manual concatenation.

**④ Grid search ready.** `GridSearchCV` and `cross_val_score` work directly with
a Pipeline. You can tune hyperparameters from any step (e.g., the `k` in
`SelectKBest` or the `max_depth` of the tree) with a single call.

---

## 🐛 Bug Encountered & Fix

### The Error

```
ValueError: could not convert string to float: 'male'
```

The pipeline crashed at `pipe.fit(x_train, y_train)`, specifically inside
`trf4` (`SelectKBest`), which only accepts numeric data. The `Sex` column —
containing strings `'male'`/`'female'` — was never one-hot encoded, and
slipped all the way through to `trf4` as raw text.

### Root Cause — Column Index Drift

The bug originated in `trf2`, not `trf4`. This is what makes it tricky: the
mistake is made silently two steps earlier, and only explodes later.

When `ColumnTransformer` uses `remainder='passthrough'`, it **reorders the
output columns**: the explicitly-specified columns come out first, followed by
the remaining (passthrough) columns in their original order.

After `trf1` — which specified `Age` (index 2) and `Embarked` (index 6) for
imputation — the output column order changed:

**Input to trf1 (original order):**

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|-------|---|---|---|---|---|---|---|
| Column | Pclass | Sex | Age | SibSp | Parch | Fare | Embarked |

**Output of trf1 (reordered by ColumnTransformer):**

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|-------|---|---|---|---|---|---|---|
| Column | Age | Embarked | Pclass | **Sex** | SibSp | Parch | Fare |

`trf2` was written with the original indices `[1, 6]` — which were
`Sex` and `Embarked` in the original DataFrame. But after `trf1`, index `1`
is now `Embarked` and index `6` is `Fare`. So `trf2` encoded
**Embarked and Fare** instead of **Sex and Embarked**. `Sex` was passed through
as raw strings with no error or warning.

### The Fix

Update `trf2` to use the **post-trf1 column positions**:

```python
# ❌ Bug — uses original indices, before trf1 reshuffles the columns
trf2 = ColumnTransformer([
    ("ohe_sex_embarked", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
     [1, 6])   # index 1 = Embarked (not Sex!), index 6 = Fare (not Embarked!)
], remainder="passthrough")

# ✅ Fix — uses post-trf1 indices: Sex is now at 3, Embarked is now at 1
trf2 = ColumnTransformer([
    ("ohe_sex_embarked", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
     [1, 3])
], remainder="passthrough")
```

### Better Long-Term Fix — Use Column Names

Hardcoded integer indices are inherently fragile across chained
`ColumnTransformer` steps. The robust solution is `.set_output(transform="pandas")`,
which makes each transformer output a DataFrame with column names preserved:

```python
trf1 = ColumnTransformer([
    ("impute_age",      SimpleImputer(),                         ["Age"]),
    ("impute_embarked", SimpleImputer(strategy="most_frequent"), ["Embarked"])
], remainder="passthrough").set_output(transform="pandas")

trf2 = ColumnTransformer([
    ("ohe_sex_embarked", OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
     ["Sex", "Embarked"])
], remainder="passthrough").set_output(transform="pandas")
```

Now `trf2` finds `Sex` and `Embarked` by name — regardless of what order
`trf1` produced them in. No manual index tracking needed at any step.

---

## 📊 Pipeline Architecture (Visual Summary)

```
x_train (712 rows × 7 cols)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  trf1 — ColumnTransformer (Imputation)                      │
│  • SimpleImputer(mean)          → fills Age (137 missing)   │
│  • SimpleImputer(most_frequent) → fills Embarked (2 miss.)  │
│  Output: 712 × 7  (all values filled, columns reordered)   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  trf2 — ColumnTransformer (One-Hot Encoding)                │
│  • OHE(Sex)      → 2 binary cols  [female, male]           │
│  • OHE(Embarked) → 3 binary cols  [C, Q, S]               │
│  Output: 712 × 10 (5 orig numeric + 2 sex + 3 embarked)    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  trf3 — ColumnTransformer (MinMaxScaler)                    │
│  • Scales all 10 features to [0, 1]                        │
│  • Required for chi2 (no negative values allowed)           │
│  Output: 712 × 10 (same shape, values in [0,1])            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  trf4 — SelectKBest(chi2, k=8)                              │
│  • Scores all 10 features by chi-squared statistic         │
│  • Keeps top 8, drops 2 least significant                  │
│  Output: 712 × 8                                           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  trf5 — DecisionTreeClassifier                              │
│  • Trains on the 8 selected features                       │
│  Output: fitted model                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 How to Run

### 1. Mount Google Drive (Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install dependencies (all are pre-installed in Colab)
```python
# No additional installs needed
# numpy, pandas, scikit-learn, seaborn, matplotlib are available by default
```

### 3. Run the notebooks in order
- Run `without_pipeline.ipynb` first to understand the raw preprocessing logic
- Then run `Titanic_by_pipelining.ipynb` to see the same logic in Pipeline form

### 4. Save the trained pipeline
```python
import joblib
joblib.dump(pipe, 'titanic_pipeline.pkl')

# Load and predict on new data later
pipe = joblib.load('titanic_pipeline.pkl')
pipe.predict(new_data)
```

---

## 🧠 Key Takeaways

**1. Pipelines prevent data leakage by construction.**
With manual preprocessing, it's easy to accidentally call `fit_transform` on
test data. A Pipeline handles the `fit` vs `transform` distinction automatically.

**2. `ColumnTransformer` with `remainder='passthrough'` reorders columns.**
Specified columns come first, passthrough columns come after — in their original
order. Hardcoded integer indices in a downstream step will point to wrong columns.
Always verify indices after each `ColumnTransformer` step, or switch to column
names with `.set_output(transform="pandas")`.

**3. Silent bugs are the most dangerous kind.**
The `Sex` column slipped through `trf2` unencoded with no error or warning.
The crash only happened two steps later at `trf4`. When debugging a pipeline,
always trace the error back to the actual source, not just where it explodes.

**4. A high train accuracy with low test accuracy means overfitting.**
The Decision Tree hit 98.3% on training data but only 60.9% on test — a clear
sign it memorized the training set. Regularization (`max_depth`, `min_samples_leaf`)
or a better ensemble model would help.

**5. Pipelines are deployment-ready objects.**
A fitted `pipe` can be serialized with `joblib` and loaded anywhere. It applies
all preprocessing and predicts in a single call — no manual reconstruction of
the transformation steps needed.

---

## 📁 File Structure

```
📦 Titanic - Pipelining
 ┣ 📓 without_pipeline.ipynb      ← Manual preprocessing approach
 ┣ 📓 Titanic_by_pipelining.ipynb ← sklearn Pipeline approach (this file)
 ┣ 📄 train.csv                   ← Titanic training data
 ┗ 📄 README.md                   ← You are here
```

---

## 📚 References

- [sklearn Pipeline docs](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [sklearn ColumnTransformer docs](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)
- [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic)
- [sklearn set_output API](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html)
