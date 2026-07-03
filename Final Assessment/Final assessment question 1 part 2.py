# SpaceX Launch Analysis Project
# NumPy + Pandas + 10 Seaborn Maps + 5 Machine Learning Models

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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


try:
    BASE_DIR = Path(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Final Assessment\spacex_launches.csv').resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_PATH = BASE_DIR / "spacex_launches.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("Project Folder:", BASE_DIR)
print("Dataset Path:", DATA_PATH)


df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Final Assessment\spacex_launches.csv')

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())


data = df.copy()

data["date_utc"] = pd.to_datetime(data["date_utc"], errors="coerce")

data["year"] = data["date_utc"].dt.year
data["month"] = data["date_utc"].dt.month
data["day"] = data["date_utc"].dt.day
data["dayofweek"] = data["date_utc"].dt.dayofweek
data["hour"] = data["date_utc"].dt.hour

data["landing_type_filled"] = data["landing_type"].fillna("No Landing")

data["flight_number_log"] = np.log1p(data["flight_number"])

data["has_crew"] = np.where(
    data["crew_count"] > 0,
    1,
    0
)

data["multiple_payloads"] = np.where(
    data["payloads_count"] > 1,
    1,
    0
)

data["launch_era"] = np.select(
    [
        data["year"] < 2010,
        data["year"].between(2010, 2019),
        data["year"] >= 2020
    ],
    [
        "Before 2010",
        "2010-2019",
        "2020 and After"
    ],
    default="Unknown"
)

data["success"] = data["success"].astype(int)

print("\nData Cleaning Completed")
print(data.head())


def remove_legend():
    legend = plt.gca().get_legend()
    if legend is not None:
        legend.remove()


sns.set_theme(style="whitegrid")


missing_df = data.isnull().sum().reset_index()
missing_df.columns = ["Column", "Missing Values"]
missing_df = missing_df.sort_values("Missing Values", ascending=False)

plt.figure(figsize=(11, 6))
sns.barplot(
    data=missing_df,
    x="Column",
    y="Missing Values",
    hue="Column",
    dodge=False
)
remove_legend()
plt.title("Map 1: Missing Values in Each Column")
plt.xlabel("Columns")
plt.ylabel("Missing Values")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_missing_values.png", dpi=300, bbox_inches="tight")
plt.close()


numeric_cols = [
    "flight_number",
    "success",
    "crew_count",
    "payloads_count",
    "cores_reused",
    "landing_success",
    "year",
    "month",
    "day",
    "dayofweek",
    "hour",
    "flight_number_log",
    "has_crew",
    "multiple_payloads"
]

plt.figure(figsize=(11, 7))
corr = data[numeric_cols].corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Map 2: Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()


plt.figure(figsize=(7, 5))
sns.countplot(
    data=data,
    x="success",
    hue="success"
)
remove_legend()
plt.title("Map 3: Mission Success vs Failure Count")
plt.xlabel("Mission Success: 0 = Failure, 1 = Success")
plt.ylabel("Number of Launches")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_success_failure_count.png", dpi=300, bbox_inches="tight")
plt.close()


plt.figure(figsize=(9, 5))
sns.countplot(
    data=data,
    x="rocket_name",
    hue="success"
)
plt.title("Map 4: Mission Outcome by Rocket Name")
plt.xlabel("Rocket Name")
plt.ylabel("Number of Launches")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_outcome_by_rocket_name.png", dpi=300, bbox_inches="tight")
plt.close()


plt.figure(figsize=(10, 5))
sns.countplot(
    data=data,
    x="launchpad_name",
    hue="success"
)
plt.title("Map 5: Mission Outcome by Launchpad")
plt.xlabel("Launchpad Name")
plt.ylabel("Number of Launches")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_outcome_by_launchpad.png", dpi=300, bbox_inches="tight")
plt.close()


yearly_launches = (
    data.groupby("year", as_index=False)
    .size()
    .rename(columns={"size": "launch_count"})
)

plt.figure(figsize=(11, 5))
sns.lineplot(
    data=yearly_launches,
    x="year",
    y="launch_count",
    marker="o"
)
plt.title("Map 6: Number of Launches by Year")
plt.xlabel("Year")
plt.ylabel("Launch Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_launch_count_by_year.png", dpi=300, bbox_inches="tight")
plt.close()


success_by_year = (
    data.groupby("year", as_index=False)["success"]
    .mean()
    .rename(columns={"success": "success_rate"})
)

plt.figure(figsize=(11, 5))
sns.lineplot(
    data=success_by_year,
    x="year",
    y="success_rate",
    marker="o"
)
plt.title("Map 7: Success Rate by Year")
plt.xlabel("Year")
plt.ylabel("Success Rate")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_success_rate_by_year.png", dpi=300, bbox_inches="tight")
plt.close()


success_by_rocket = (
    data.groupby("rocket_name", as_index=False)["success"]
    .mean()
    .rename(columns={"success": "success_rate"})
    .sort_values("success_rate", ascending=False)
)

plt.figure(figsize=(9, 5))
sns.barplot(
    data=success_by_rocket,
    x="rocket_name",
    y="success_rate",
    hue="rocket_name",
    dodge=False
)
remove_legend()
plt.title("Map 8: Success Rate by Rocket Name")
plt.xlabel("Rocket Name")
plt.ylabel("Success Rate")
plt.ylim(0, 1.05)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_success_rate_by_rocket.png", dpi=300, bbox_inches="tight")
plt.close()


plt.figure(figsize=(8, 5))
sns.boxplot(
    data=data,
    x="success",
    y="payloads_count"
)
plt.title("Map 9: Payload Count Distribution by Mission Outcome")
plt.xlabel("Mission Success: 0 = Failure, 1 = Success")
plt.ylabel("Payload Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_payload_count_by_outcome.png", dpi=300, bbox_inches="tight")
plt.close()


plt.figure(figsize=(9, 5))
sns.countplot(
    data=data,
    x="landing_type_filled",
    hue="success"
)
plt.title("Map 10: Landing Type by Mission Outcome")
plt.xlabel("Landing Type")
plt.ylabel("Number of Launches")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_landing_type_by_outcome.png", dpi=300, bbox_inches="tight")
plt.close()

print("\n10 Seaborn maps saved in the outputs folder.")


y = data["success"]

feature_columns = [
    "flight_number",
    "flight_number_log",
    "rocket_name",
    "launchpad_name",
    "crew_count",
    "payloads_count",
    "cores_reused",
    "landing_type_filled",
    "year",
    "month",
    "day",
    "dayofweek",
    "hour",
    "has_crew",
    "multiple_payloads",
    "launch_era"
]

X = data[feature_columns]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()


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
        max_depth=4,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=7
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )
}


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


results = []

for model_name, model in models.items():

    print("\n" + "=" * 80)
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
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_accuracy_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    cv_balanced_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="balanced_accuracy"
    )

    results.append({
        "Model": model_name,
        "Test Accuracy (%)": round(test_accuracy * 100, 2),
        "Balanced Accuracy (%)": round(balanced_acc * 100, 2),
        "Cross Validation Accuracy (%)": round(cv_accuracy_scores.mean() * 100, 2),
        "Cross Validation Balanced Accuracy (%)": round(cv_balanced_scores.mean() * 100, 2)
    })

    print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
    print("Balanced Accuracy:", round(balanced_acc * 100, 2), "%")
    print("Cross Validation Accuracy:", round(cv_accuracy_scores.mean() * 100, 2), "%")
    print("Cross Validation Balanced Accuracy:", round(cv_balanced_scores.mean() * 100, 2), "%")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("Final Model Accuracy Results:")
print(results_df)

results_path = OUTPUT_DIR / "model_accuracy_results.csv"
results_df.to_csv(results_path, index=False)

print("\nModel accuracy results saved at:", results_path)


plt.figure(figsize=(12, 6))
sns.barplot(
    data=results_df,
    x="Model",
    y="Test Accuracy (%)",
    hue="Model",
    dodge=False
)
remove_legend()
plt.title("Machine Learning Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_model_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.close()


print("\nFinal assessment question 1 part 2 completed!!!!")
