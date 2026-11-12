# Telco Customer Churn — Supervised Learning

## What this project does

`Telco_Churn_Analysis.ipynb` is a complete, reproducible binary-classification experiment for predicting whether a telecom customer will churn. It uses the IBM Telco Customer Churn CSV supplied with this project. The original CSV is never changed.

## Run it in Google Colab

1. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to the top level of Google Drive (`My Drive`).
2. Upload/open `Telco_Churn_Analysis.ipynb` in Google Colab.
3. Run cells from top to bottom, or select **Runtime → Run all**.
4. Approve the Google Drive mount when asked. The notebook expects `/content/drive/MyDrive/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
5. Download `outputs/` and `models/` from Colab’s Files panel if you want to retain them after the runtime closes.

The first code cell installs `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `xgboost`, and `joblib`. A CPU runtime is sufficient.

## Notebook, cell by cell

1. **Title and introduction** — Defines churn prediction as a binary supervised-learning task and states the objectives.
2. **Environment setup** — Installs/imports dependencies, fixes the random seed, configures plots, and creates artifact folders.
3. **Dataset-loading explanation** — Explains the expected IBM dataset structure and Drive location.
4. **Dataset loading and verification** — Mounts Drive, reads the CSV, verifies expected columns, and displays records, types, statistics, and a feature table.
5. **EDA explanation** — Introduces data quality, distributions, outliers, and correlation analysis.
6. **EDA** — Converts `TotalCharges`, checks missing values and duplicates, generates charts, and prints dataset-derived churn observations.
7. **Preprocessing explanation** — Explains identifier removal, splitting, and leakage prevention.
8. **Preprocessing** — Removes `customerID`, encodes the target, makes a stratified 70:30 split, and defines imputation, one-hot encoding, scaling, and the `ColumnTransformer`.
9. **Models and imbalance explanation** — Covers all five algorithms, ensemble learning, and SMOTE.
10. **Baseline models and SMOTE pipeline** — Defines all five estimators, trains the three baseline models, and displays initial metrics and confusion matrices.
11. **Hyperparameter-tuning explanation** — Explains five-fold stratified cross-validation and F1 optimization without test-set tuning.
12. **Grid search** — Tunes all five models, saves the best parameters/CV scores, and retains best estimators.
13. **Evaluation explanation** — Defines confusion-matrix outcomes, precision, recall, F1, ROC-AUC, and precision–recall curves.
14. **Final evaluation** — Evaluates tuned models on untouched test data, saves result files, and creates comparison, ROC/PR, and confusion-matrix charts.
15. **Interpretability explanation** — Explains permutation/tree importance and association versus causation.
16. **Feature importance** — Saves raw-feature permutation importance and charts; also generates tree importance when available.
17. **Final-selection explanation** — Explains selecting and reusing a fitted pipeline.
18. **Save and prediction check** — Saves/reloads the best pipeline, tests raw inputs, checks predictions/probabilities, and saves sample predictions.

## Generated artifacts

After execution, outputs are separated by type:

```text
outputs/
├── images/                         # PNG graphs
│   ├── numeric_distributions.png
│   ├── categorical_churn.png
│   ├── outliers.png
│   ├── correlation.png
│   ├── model_comparison.png
│   ├── roc_pr_curves.png
│   ├── final_confusion_matrices.png
│   ├── permutation_importance.png
│   └── tree_feature_importance.png # When available
└── results/                        # Separate calculated result files
    ├── hyperparameter_search_results.csv
    ├── evaluation_results.csv
    ├── model_comparison.csv
    ├── classification_reports.txt
    ├── results_summary.txt
    ├── permutation_importance.csv
    └── sample_prediction_check.csv

models/
└── best_churn_model.pkl             # Fitted preprocessing + classifier pipeline
```

All results are computed during your own run; no model performance is pre-filled or fabricated.
