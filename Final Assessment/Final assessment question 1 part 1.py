# SpaceX Machine Learning Project
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    BASE_DIR = Path(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Final Assessment\database.csv').resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_PATH = BASE_DIR / "database.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Project Folder:", BASE_DIR)
print("Dataset Path:", DATA_PATH)

df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Final Assessment\database.csv')

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

df_clean = df.copy()

# Convert launch date
df_clean["Launch Date Parsed"] = pd.to_datetime(
    df_clean["Launch Date"],
    errors="coerce"
)

df_clean["Year"] = df_clean["Launch Date Parsed"].dt.year
df_clean["Month"] = df_clean["Launch Date Parsed"].dt.month
df_clean["DayOfWeek"] = df_clean["Launch Date Parsed"].dt.dayofweek

# Convert launch time
df_clean["Launch Time Parsed"] = pd.to_datetime(
    df_clean["Launch Time"],
    errors="coerce"
)

df_clean["LaunchHour"] = df_clean["Launch Time Parsed"].dt.hour

# Clean target column
df_clean["Mission Outcome Clean"] = (
    df_clean["Mission Outcome"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Convert target into 0 and 1
df_clean["Mission_Target"] = np.where(
    df_clean["Mission Outcome Clean"].str.contains("success"),
    1,
    0
)

# NumPy transformation
df_clean["Payload_Mass_Log"] = np.log1p(df_clean["Payload Mass (kg)"])

# Payload mass category using NumPy
conditions = [
    df_clean["Payload Mass (kg)"].isna(),
    df_clean["Payload Mass (kg)"] < 1000,
    df_clean["Payload Mass (kg)"].between(1000, 4000),
    df_clean["Payload Mass (kg)"] > 4000
]

choices = ["Unknown", "Light", "Medium", "Heavy"]

df_clean["Mass_Category"] = np.select(
    conditions,
    choices,
    default="Unknown"
)

print("\nData Cleaning Completed")
print(df_clean[["Mission Outcome", "Mission_Target", "Payload Mass (kg)", "Mass_Category"]].head())

sns.set_theme(style="whitegrid")

numeric_columns = [
    "Payload Mass (kg)",
    "Payload_Mass_Log",
    "Year",
    "Month",
    "DayOfWeek",
    "LaunchHour",
    "Mission_Target"
]

plt.figure(figsize=(9, 6))

correlation = df_clean[numeric_columns].corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Graph 1: Correlation Heatmap")
plt.tight_layout()

heatmap_path = OUTPUT_DIR / "01_correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", heatmap_path)

plt.figure(figsize=(11, 6))

sns.countplot(
    data=df_clean,
    x="Vehicle Type",
    hue="Mission Outcome"
)

plt.title("Graph 2: Mission Outcome by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Number of Missions")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

vehicle_graph_path = OUTPUT_DIR / "02_mission_by_vehicle_type.png"
plt.savefig(vehicle_graph_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", vehicle_graph_path)

success_by_site = (
    df_clean
    .groupby("Launch Site", as_index=False)["Mission_Target"]
    .mean()
    .rename(columns={"Mission_Target": "Success_Rate"})
    .sort_values("Success_Rate", ascending=False)
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=success_by_site,
    x="Launch Site",
    y="Success_Rate"
)

plt.title("Graph 3: Success Rate by Launch Site")
plt.xlabel("Launch Site")
plt.ylabel("Success Rate")
plt.ylim(0, 1)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

site_graph_path = OUTPUT_DIR / "03_success_rate_by_launch_site.png"
plt.savefig(site_graph_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", site_graph_path)

# These columns are used for prediction
feature_columns = [
    "Launch Site",
    "Vehicle Type",
    "Payload Type",
    "Payload Mass (kg)",
    "Payload Orbit",
    "Customer Type",
    "Customer Country",
    "Year",
    "Month",
    "DayOfWeek",
    "LaunchHour",
    "Payload_Mass_Log",
    "Mass_Category"
]

X = df_clean[feature_columns]
y = df_clean["Mission_Target"]

numeric_features = [
    "Payload Mass (kg)",
    "Year",
    "Month",
    "DayOfWeek",
    "LaunchHour",
    "Payload_Mass_Log"
]

categorical_features = [
    column for column in feature_columns
    if column not in numeric_features
]


# OneHotEncoder compatibility for old and new sklearn versions
try:
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
except TypeError:
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )


numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", one_hot_encoder)
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
}

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

results = []

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("Training Model:", model_name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    results.append({
        "Model": model_name,
        "Test Accuracy (%)": round(test_accuracy * 100, 2),
        "Cross Validation Accuracy (%)": round(cv_scores.mean() * 100, 2)
    })

    print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
    print("Cross Validation Accuracy:", round(cv_scores.mean() * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("Final Accuracy Results")
print(results_df)

accuracy_path = OUTPUT_DIR / "model_accuracy_results.csv"
results_df.to_csv(accuracy_path, index=False)

print("\nAccuracy results saved at:", accuracy_path)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Test Accuracy (%)"
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

accuracy_graph_path = OUTPUT_DIR / "04_model_accuracy_comparison.png"
plt.savefig(accuracy_graph_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", accuracy_graph_path)

print("\nFinal assessment part 1 completed!!!")

# This code cleans and analyzes SpaceX launch data, then creates charts such as a correlation heatmap,
# mission outcomes by vehicle type, and success rate by launch site.

# It trains and compares Logistic Regression, Decision Tree, and Random Forest models
# to predict whether a SpaceX mission will be successful.
