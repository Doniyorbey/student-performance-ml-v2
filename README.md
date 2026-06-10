# Explainable Student Risk Analytics

A Pearson BTEC Level 6 applied machine-learning capstone prototype for early identification of students at risk of failing. The project uses the UCI Student Performance dataset and is deployed as a Streamlit application.

## Critical project boundary

- The dataset is **not** PDP University student data.
- The application is an **academic prototype**, not a production decision system.
- Predictions must not be used as the sole basis for sanctions, exclusion or admissions decisions.
- SHAP values explain the fitted model; they do not establish causality.

## Methodological design

The target is coded as:

- `0 = Fail / at risk` when `G3 < 10`
- `1 = Pass` when `G3 >= 10`

`G1`, `G2` and `G3` are excluded from model inputs to prevent the system from merely reproducing prior/final grades.

### Why Macro-F1 is the primary metric

The Pass class is larger in the source dataset. Optimising ordinary positive-class F1 can make a model that mainly predicts Pass appear strong while it misses at-risk students. The application therefore:

- selects hyperparameters using **Macro-F1**;
- reports **Risk Recall**, **F1-Risk**, **Missed Risk Rate**, **Balanced Accuracy**, **MCC** and **risk PR-AUC**;
- compares every model with a **Dummy Baseline**.

### Validation

- Quick mode: 3 outer folds × 3 inner folds
- Full mode: 5 outer folds × 5 inner folds
- Preprocessing occurs inside every cross-validation fold
- Outer-fold predictions are retained for ROC, PR, calibration, fairness and error analysis

## Models

- Dummy Baseline
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM RBF
- KNN

## Application evidence

The app includes:

- data-quality audit and dataset fingerprint;
- leakage-safe preprocessing;
- nested cross-validation and GridSearchCV;
- Optuna optimisation;
- balanced and risk-focused model comparison;
- ROC, risk Precision–Recall and threshold analysis;
- calibration curve and Brier score;
- bootstrap confidence intervals and Wilcoxon comparison;
- missed-risk and false-alarm error analysis;
- subgroup/fairness diagnostics;
- global and individual SHAP explanations;
- human-reviewed individual prediction;
- model card, validity assessment and downloadable evidence pack.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The repository root must contain `student-mat.csv`.

## Streamlit deployment

- Repository: your GitHub repository
- Branch: `main`
- Main file path: `app.py`

## BTEC Level 6 contribution

The application supplies strong technical evidence for implementation, data analysis, comparison of patterns, professional communication and evaluation of validity/reliability. It does **not** replace the written critical work required for alternative research directions, project-management evaluation and personal/professional reflection.
