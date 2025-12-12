# ============================================================
# PROJECT: Customer Churn Prediction (IBM Telco Dataset)
# Clean Version With:
# - Missing Value Handling
# - SMOTE for Imbalance
# - 5 ML Models
# - Hyperparameter Tuning (RF + XGB)
# - ROC Curves + SHAP
# ============================================================
import matplotlib
matplotlib.use("TkAgg")
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, RocCurveDisplay
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. Load Dataset
# ============================================================

DATA_PATH = os.path.join("data", "Telco_customer_churn.csv")
df = pd.read_csv(DATA_PATH)
print(df.head())

# Fix numeric fields (convert strings like ' ' → NaN)
numeric_fields = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Churn Score",
    "CLTV"
]

for col in numeric_fields:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# 2. Define Target + Drop Columns
# ============================================================

drop_cols = [
    "Churn Value",
    "Churn Label",
    "Churn Score",
    "Churn Reason",
    "CustomerID",
    "Count",
    "Lat Long",
    "Zip Code",
]

X = df.drop(drop_cols, axis=1)
y = df["Churn Value"]

# ============================================================
# 3. Feature Groups
# ============================================================

numeric_features = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
ordinal_features = ["Contract"]
ordinal_categories = [["Month-to-month", "One year", "Two year"]]

nominal_features = X.select_dtypes(include=["object"]).columns.tolist()
nominal_features = [c for c in nominal_features if c not in ordinal_features]

# ============================================================
# 4. Preprocessing Pipelines
# ============================================================

numeric_transformer = ImbPipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

ordinal_transformer = ImbPipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(categories=ordinal_categories))
])

nominal_transformer = ImbPipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("ord", ordinal_transformer, ordinal_features),
        ("cat", nominal_transformer, nominal_features),
    ]
)

# ============================================================
# 5. Models to Evaluate
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

results = {}

# ============================================================
# 6. Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 7. Train All Models with SMOTE
# ============================================================

for name, model in models.items():

    pipe = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": pipe,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob)
    }

    print(f"\n===========================\n{name}\n===========================")
    print("Accuracy:", results[name]["accuracy"])
    print("ROC-AUC:", results[name]["auc"])
    print(classification_report(y_test, y_pred))

# ============================================================
# 8. Hyperparameter Tuning - Random Forest
# ============================================================

rf_pipe = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier())
])

rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    rf_pipe,
    rf_params,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

print("\nBest RF Params:", rf_grid.best_params_)
print("Best RF AUC:", rf_grid.best_score_)

results["Random Forest (Tuned)"] = {
    "model": best_rf,
    "accuracy": None,
    "auc": rf_grid.best_score_
}

# ============================================================
# 9. Hyperparameter Tuning - XGBoost
# ============================================================

xgb_pipe = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(eval_metric="logloss"))
])

xgb_params = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [4, 6],
    "model__learning_rate": [0.01, 0.05],
    "model__subsample": [0.7, 1.0],
    "model__colsample_bytree": [0.7, 1.0]
}

xgb_grid = GridSearchCV(
    xgb_pipe,
    xgb_params,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_

print("\nBest XGB Params:", xgb_grid.best_params_)
print("Best XGB AUC:", xgb_grid.best_score_)

results["XGBoost (Tuned)"] = {
    "model": best_xgb,
    "accuracy": None,
    "auc": xgb_grid.best_score_
}

# ============================================================
# 10. Plot ROC for Every Model
# ============================================================

plt.figure(figsize=(8, 6))

for name, r in results.items():
    RocCurveDisplay.from_estimator(r["model"], X_test, y_test, name=name)

plt.title("ROC Curves for All Models")

print("ROC curve saved to Desktop as roc_curves.png")

# ============================================================
# 11. SHAP on Tuned XGBoost
# ============================================================

print("\nRunning SHAP (may take 20–40 seconds)...")

xgb_internal = best_xgb.named_steps["model"]
X_processed = best_xgb.named_steps["preprocessor"].transform(X)

explainer = shap.TreeExplainer(xgb_internal)
shap_values = explainer.shap_values(X_processed)

shap.summary_plot(shap_values, X_processed,
                  feature_names=best_xgb.named_steps["preprocessor"].get_feature_names_out())

