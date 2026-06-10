# Explainable Student Risk Analytics

A fully English Pearson BTEC Level 6 applied-AI capstone prototype for early identification of students at risk of failing. The project uses the UCI Student Performance dataset and is deployed as a Streamlit application.

## Project boundary

- The dataset is **not** PDP University student data.
- The application is an **academic prototype**, not a production decision system.
- Predictions must not be used as the sole basis for sanctions, exclusion, admissions or grading decisions.
- SHAP values explain the fitted model; they do not establish causality.
- External validation is required before use in another institution or country.

## Research design

The target is coded as:

- `0 = Fail / at risk` when `G3 < 10`
- `1 = Pass` when `G3 >= 10`

`G1`, `G2` and `G3` are excluded from model inputs so that the system operates as a genuine early-warning prototype rather than reproducing prior or final grades.

### Primary metric

The Pass class is larger in the source dataset. Optimising ordinary positive-class F1 can make a model appear strong while it misses at-risk students. The application therefore:

- selects hyperparameters using **Macro-F1**;
- reports **Risk Recall**, **F1-Risk**, **Missed Risk Rate**, **Balanced Accuracy**, **MCC** and **risk PR-AUC**;
- compares every model with a **Dummy Baseline**;
- retains out-of-fold predictions for all reliability, fairness and error analyses.

### Validation

- Quick mode: 3 outer folds × 3 inner folds
- Full mode: 5 outer folds × 5 inner folds
- Preprocessing occurs inside every cross-validation fold
- The inner loop selects hyperparameters
- The untouched outer loop estimates generalisation
- A final tuned pipeline is fitted after evaluation

## Candidate models

- Dummy Baseline
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM with RBF kernel
- K-Nearest Neighbours

## Application evidence

The application includes:

- a data-quality audit and reproducible dataset fingerprint;
- leakage-safe imputation, scaling and one-hot encoding;
- nested cross-validation and GridSearchCV;
- optional Optuna optimisation;
- balanced and risk-focused model comparison;
- ROC and risk Precision–Recall curves;
- threshold and confusion-matrix analysis;
- cost-sensitive threshold recommendation;
- calibration curve and Brier score;
- bootstrap confidence intervals;
- Friedman omnibus testing;
- pairwise Wilcoxon tests with Holm correction;
- missed-risk and false-alarm analysis;
- learning-curve diagnosis;
- subgroup and fairness diagnostics;
- global and individual SHAP explanations;
- human-reviewed individual prediction;
- a model card, validity assessment and downloadable evidence pack;
- an automatically generated executive summary.

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

## Recommended final workflow

1. Run **Quick mode** to verify every page.
2. Test Results, Fairness, Validity, SHAP and Prediction.
3. Run **Full mode** once for final evidence.
4. Download the evidence pack and executive summary.
5. Preserve the exported files for the report and viva.

## BTEC Level 6 contribution

The application provides strong technical evidence for implementation, data analysis, comparison of patterns, professional communication and evaluation of validity and reliability. It does not replace the critical written work required for alternative research directions, project-management evaluation and reflective personal development.

## Key academic foundations

- Cortez, P. and Silva, A.M.G. (2008) *Using data mining to predict secondary school student performance*.
- Varma, S. and Simon, R. (2006) ‘Bias in error estimation when using cross-validation for model selection’, *BMC Bioinformatics*, 7, 91.
- Pedregosa, F. et al. (2011) ‘Scikit-learn: Machine Learning in Python’, *Journal of Machine Learning Research*, 12, pp. 2825–2830.
- Lundberg, S.M. and Lee, S.-I. (2017) ‘A Unified Approach to Interpreting Model Predictions’, *Advances in Neural Information Processing Systems*, 30.
- Mitchell, M. et al. (2019) ‘Model Cards for Model Reporting’, *Proceedings of the Conference on Fairness, Accountability, and Transparency*.
