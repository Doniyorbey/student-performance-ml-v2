Student Performance ML
BTEC Level 6 Independent Project

Project Title
Machine Learning Model Optimisation for Student Academic Risk Prediction

Student
Eltezarov Doniyorbek
Group: 22-305
University: PDP University
Academic Year: 2025-2026

1. Project Overview
This project is a machine-learning prototype for early identification of students who may be at academic risk. The system uses the UCI Student Performance dataset and predicts whether a student is likely to pass or be at risk/fail.

The project focuses on:
- dataset audit and exploratory data analysis;
- leakage-safe preprocessing;
- model training and comparison;
- hyperparameter optimisation;
- nested cross-validation;
- risk-focused evaluation metrics;
- fairness diagnostics;
- SHAP-based explainability;
- single-student prediction;
- evidence pages for BTEC assessment.

The Streamlit interface is used to demonstrate the model workflow and results. The main academic focus is model development, optimisation, validation and critical evaluation.

2. Programming Languages
- Python

3. Frameworks and Libraries
Main technologies used:
- Streamlit: web application interface
- pandas: data loading and data preparation
- numpy: numerical operations
- scikit-learn: preprocessing, models, cross-validation and evaluation
- Plotly: interactive charts
- Matplotlib: visualisation support
- SHAP: model explainability
- joblib: model/object persistence where required

4. Dataset / Database
Database type:
- CSV dataset file

Dataset file:
- student-mat.csv

No SQL database is used in this project. The dataset is stored as a CSV file and loaded directly by the Python application.

5. Source Code Structure
The final ZIP package should contain:

Source_Code/
├── frontend/
│   └── app.py
├── backend/
│   └── ml_pipeline.py
└── database/
    └── student-mat.csv

File descriptions:
- app.py: Streamlit user interface and application navigation
- ml_pipeline.py: machine-learning pipeline, preprocessing, training and evaluation logic
- student-mat.csv: dataset used by the project
- requirements.txt: Python dependencies required to run the project

6. How to Run the Project Locally
Step 1: Install Python 3.10 or newer.

Step 2: Open the project folder in VS Code or a terminal.

Step 3: Install dependencies:
pip install -r requirements.txt

Step 4: Make sure the dataset file is available:
student-mat.csv

Step 5: Run the Streamlit application:
streamlit run app.py

Step 6: Open the local URL displayed in the terminal. Usually:
http://localhost:8501

7. Online Deployment
The project is deployed using Streamlit Community Cloud.

Live app URL:
https://student-performance-ml-v2-xndaouobmzgsqn5sssibdq.streamlit.app/

GitHub repository:
https://github.com/Doniyorbey/student-performance-ml-v2

8. Main Application Modules
The application includes the following main pages:

Overview
Introduces the project purpose and modelling workflow.

EDA & Data Audit
Displays target distribution, grade distribution, dataset preview, correlations and data-quality checks.

Model Training
Runs nested cross-validation and compares models using risk-focused metrics.

Results
Shows model comparison, best model, Macro-F1, Risk Recall and related performance results.

Fairness Analysis
Checks subgroup-level performance differences.

SHAP Values
Explains important model features and model behaviour.

Prediction
Allows a user to enter a student profile and generate a prediction.

Model Card / Research Evidence
Summarises model purpose, intended use, limitations and project evidence.

9. Models Used
The project compares:
- Dummy Baseline
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM
- KNN

10. Key Evaluation Metrics
The project uses:
- Macro-F1
- Risk Recall
- F1-Risk
- Accuracy
- Balanced Accuracy
- ROC-AUC
- Matthews Correlation Coefficient
- Confusion Matrix

Macro-F1 is used as the primary metric because accuracy alone can be misleading when the pass class is larger than the at-risk class.

11. Important Methodological Notes
- G1, G2 and G3 grade columns are excluded from model input features.
- G3 is used only to define the pass/fail target.
- The project avoids using final or prior-period grades as predictors to keep the early-warning scenario more realistic.
- Nested cross-validation is used to reduce over-optimistic model evaluation.
- The system is designed as a decision-support prototype, not as an automatic decision-making system.

12. Limitations
- The dataset is public and does not represent PDP University students.
- The model should not be used for real academic decisions without local validation.
- Predictions should be reviewed by a human adviser or academic staff member.
- The project is intended for academic demonstration and BTEC assessment.

13. Submission Package
The final submission ZIP should include:
- Report.docx
- Presentation.pptx
- Source_Code folder
- User_Manual.pdf
- Website_Link.txt
- README.txt

14. Contact / Author
Student: Eltezarov Doniyorbek
Group: 22-305
Project: Student Performance ML
University: PDP University
