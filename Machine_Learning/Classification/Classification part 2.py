
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier


file_path = r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\Classification\gait.csv'

if not os.path.exists(file_path):
    print("File not found. Please put gait.csv in the same folder as this Python file.")
    exit()

df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\Classification\gait.csv')

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df = df.drop_duplicates()

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

df["subject"] = df["subject"].astype(int)
df["condition"] = df["condition"].astype(int)
df["replication"] = df["replication"].astype(int)
df["leg"] = df["leg"].astype(int)
df["joint"] = df["joint"].astype(int)
df["time"] = df["time"].astype(int)


sns.set_theme(style="whitegrid")


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="condition")
plt.title("Map 1: Count of Each Condition")
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="subject", hue="condition")
plt.title("Map 2: Subject Count by Condition")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="angle", kde=True, bins=40)
plt.title("Map 3: Angle Distribution")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="condition", y="angle")
plt.title("Map 4: Angle by Condition")
plt.show()


plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="joint", y="angle")
plt.title("Map 5: Angle by Joint")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="leg", y="angle")
plt.title("Map 6: Angle by Leg")
plt.show()


line_data = df.groupby(["time", "condition"], as_index=False)["angle"].mean()

plt.figure(figsize=(12, 6))
sns.lineplot(data=line_data, x="time", y="angle", hue="condition")
plt.title("Map 7: Mean Angle Over Time by Condition")
plt.show()


plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Map 8: Correlation Heatmap")
plt.show()


plt.figure(figsize=(9, 5))
sns.barplot(data=df, x="condition", y="angle", hue="joint")
plt.title("Map 9: Mean Angle by Condition and Joint")
plt.show()


sample_df = df.sample(n=2000, random_state=42)

sns.pairplot(
    sample_df[["subject", "condition", "replication", "leg", "joint", "time", "angle"]],
    hue="condition"
)
plt.suptitle("Map 10: Pairplot of Gait Features", y=1.02)
plt.show()


pivot_df = df.pivot_table(
    index=["subject", "condition", "replication", "leg", "joint"],
    columns="time",
    values="angle",
    aggfunc="mean"
).reset_index()

pivot_df.columns = [
    f"time_{col}" if isinstance(col, (int, float, np.integer, np.floating)) else col
    for col in pivot_df.columns
]

time_cols = [col for col in pivot_df.columns if str(col).startswith("time_")]

pivot_df[time_cols] = pivot_df[time_cols].fillna(pivot_df[time_cols].median())

pivot_df["angle_mean"] = pivot_df[time_cols].mean(axis=1)
pivot_df["angle_std"] = pivot_df[time_cols].std(axis=1)
pivot_df["angle_min"] = pivot_df[time_cols].min(axis=1)
pivot_df["angle_max"] = pivot_df[time_cols].max(axis=1)
pivot_df["angle_range"] = pivot_df["angle_max"] - pivot_df["angle_min"]
pivot_df["angle_median"] = pivot_df[time_cols].median(axis=1)
pivot_df["angle_skew"] = pivot_df[time_cols].skew(axis=1)
pivot_df["angle_kurtosis"] = pivot_df[time_cols].kurtosis(axis=1)

angle_diff = pivot_df[time_cols].diff(axis=1).fillna(0)
pivot_df["mean_angle_change"] = angle_diff.abs().mean(axis=1)
pivot_df["max_angle_change"] = angle_diff.abs().max(axis=1)

print("\nFeature Engineered Dataset Shape:")
print(pivot_df.shape)

X = pivot_df.drop("condition", axis=1)
y = pivot_df["condition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, C=2.0))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
}

results = {}

print("\nMachine Learning Model Results:\n")

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("=" * 60)
    print(name)
    print("Accuracy:", round(acc * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

print("\nBest Model:")
print(best_model_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")

best_model = models[best_model_name]
best_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

accuracy_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": list(results.values())
})

accuracy_df = accuracy_df.sort_values(by="Accuracy", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=accuracy_df, x="Model", y="Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30)
plt.ylim(0, 1)
plt.show()

print("\nFinal Accuracy Comparison:")
print(accuracy_df)