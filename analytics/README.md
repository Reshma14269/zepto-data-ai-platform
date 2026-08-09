# Analytics — Module 2 (`/analytics`, 50 marks)

Full end-to-end ML pipeline on the Titanic dataset: profiling → cleaning → EDA/data story →
predictive modeling → imbalance handling → hyperparameter tuning → regression side-task → saved pipeline.

This is **one cohesive pipeline**, split across two notebooks that share a single committed CSV:

- **`01_eda.ipynb`** — loads `titanic` via `sns.load_dataset('titanic')` (the *only* network/cache load
  in the whole module), profiles it, cleans it, saves `titanic.csv`, and produces the full EDA data story
  (Part A, Tasks 1–6).
- **`02_modeling.ipynb`** — reads the same `titanic.csv`, re-applies the same cleaning, and builds/evaluates
  the classification + regression pipeline (Part B, Tasks 7–15).

## Files
- `01_eda.ipynb` — profiling, cleaning, univariate/bivariate/multivariate analysis, standardization check
- `02_modeling.ipynb` — train/test split, preprocessing pipeline, 3 classifiers, imbalance handling,
  hyperparameter tuning, regression side-task, saved pipeline
- `titanic.csv` — the one committed offline fallback for the raw dataset
- `titanic_survival_pipeline.joblib` — the final tuned Random Forest pipeline (preprocessing + estimator),
  reloadable and usable end-to-end on raw new data
- `charts/` — all saved chart images referenced in the notebooks

## Setup & how to run

```bash
cd analytics
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb    # run first — produces titanic.csv
jupyter notebook 02_modeling.ipynb   # run second — reads titanic.csv
```

`01_eda.ipynb` needs internet access the *first* time it runs (to fetch the Titanic dataset via Seaborn).
`02_modeling.ipynb` only needs the committed `titanic.csv` and does not touch the network.

## Design decisions & written interpretations

### Missing-value handling (per-column threshold rule)
| Column | % missing | Strategy | Why |
|---|---|---|---|
| `deck` | 77.22% | **Dropped column** | Too high to impute reliably; missingness tracks `pclass`/`fare` already, so an "unknown" bucket would just be a noisy proxy for a signal already present elsewhere. |
| `age` | 19.87% | **Median imputation** | Falls in the 5–30% band; median chosen over mean since `fare`-style skew analysis (Task 3) showed the data isn't symmetric, and median is more robust to outliers. |
| `embarked` / `embark_town` | 0.22% each | **Dropped rows** | Under 5%; only 2 rows affected, negligible information loss, avoids inventing a categorical value. |

### Univariate findings (Task 3)
- `age`: 65 IQR outliers, roughly bell-shaped with a mild right tail.
- `fare`: 114 IQR outliers (all high side). Mean (32.10) > median (14.45) > mode (8.05) → **strongly
  right-skewed** — a small number of high-fare (first-class) passengers pull the mean well above the
  typical fare.

### Bivariate & correlation findings (Task 4)
- Survival by sex: female **74.0%**, male **18.9%**.
- Survival by class: 1st **62.6%**, 2nd **47.3%**, 3rd **24.2%**.
- Combined: 1st-class women survived **96.7%** of the time vs. 3rd-class men at **13.5%**.
- Two strongest correlations (by |coefficient|): **`pclass` ↔ `fare`** (−0.55, better class = higher fare)
  and **`sibsp` ↔ `parch`** (0.41, family-size clustering).

### Data story (Task 5, 4 charts — see `charts/03`–`06`)
Survival was driven primarily by **sex** (women far more likely to survive), reinforced by **class/fare**
(wealthier, higher-class passengers had better odds within each sex), with **age** playing a smaller role
favoring children. Embarkation port shows a similar pattern only because it correlates with class
composition, not because of the port itself.

### Standardization check (Task 6)
`age` and `fare` z-scored to mean ≈ 0, std ≈ 1 on the full cleaned data — confirmed numerically and
visually (`charts/07`). EDA-only check; the modeling pipeline performs its own train-only scaling.

### Train/test split (Task 7)
Stratified on `survived` (train: 38.3% survived, test: 38.2% survived) because of the ~62/38 class
imbalance identified in profiling — an unstratified split risks a train/test mismatch in survival rate
that would bias evaluation.

### Preprocessing (Task 8)
`ColumnTransformer` (median-impute + scale numeric; most-frequent-impute + one-hot encode categorical)
wrapped in a `Pipeline`, fit only on the training split.

### Classifier comparison (Tasks 9–10)
*(see `02_modeling.ipynb` cell output for exact numbers — Random Forest led on accuracy and AUC, with
Logistic Regression close behind and the single Decision Tree trailing both, consistent with ensemble vs.
single-tree behavior on a dataset this size.)*

### Imbalance handling (Task 11)
Compared baseline / `class_weight='balanced'` / SMOTE (training fold only) on Random Forest. SMOTE
generally improved recall the most by balancing the training signal; `class_weight='balanced'` gave a
smaller, cheaper improvement in the same direction; the baseline had the highest precision but lowest
recall. See `02_modeling.ipynb` for the exact comparison table.

### Hyperparameter tuning (Task 12)
`GridSearchCV` over `n_estimators` / `max_depth` / `max_features` on `RandomForestClassifier(oob_score=True)`.
Best parameters found: `n_estimators=100, max_depth=None, max_features='sqrt'`. Out-of-bag score: **0.7947**.

### Regression side-task (Task 13)
Predicted `fare` from the other features with multivariate linear regression: **MAE 21.10, RMSE 41.70,
R² 0.348, Adjusted R² 0.321**. The residual plot fans out at higher predicted fares — a clear
**heteroscedasticity** pattern, meaning prediction errors are less reliable for high-fare passengers; a
log-transform of `fare` would likely fit better.

### Final model comparison & recommendation (Task 14)
Classification metrics (accuracy/precision/recall/F1/AUC) and regression metrics (MAE/RMSE/R²/Adjusted R²)
are presented as **two separate metric groups** in `02_modeling.ipynb` since they're on different scales
for different targets. **Recommendation:** deploy the **tuned Random Forest** (Task 12) as the production
classifier — it combines the strongest raw performance with a healthy OOB validation score, and can be
paired with the imbalance-handling strategy from Task 11 if minority-class recall matters for the use case.
Logistic Regression remains a good interpretable fallback; the Decision Tree is kept mainly for its visual
explainability. Linear Regression is a separate, weaker-fitting model for `fare` and isn't compared
head-to-head against the classifiers since it solves a different problem.

### Saved pipeline (Task 15)
`titanic_survival_pipeline.joblib` — the complete fitted `Pipeline` (preprocessing + tuned Random Forest),
saved via `joblib.dump`. Reloading it with `joblib.load` and predicting on raw test rows reproduces
identical predictions to the original in-memory pipeline, confirming it's usable end-to-end on raw,
unpreprocessed new data.
