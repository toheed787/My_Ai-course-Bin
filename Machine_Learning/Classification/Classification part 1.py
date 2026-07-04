import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import joblib


file_path = r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\Classification\data.csv'

if not os.path.exists(file_path):
    raise FileNotFoundError("data.csv not found. Put data.csv in the same folder as this Python file.")


output_dir = "student_project_outputs"
plot_dir = os.path.join(output_dir, "seaborn_maps")
model_dir = os.path.join(output_dir, "models")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

sns.set_theme(style="whitegrid")


print("\nLoading dataset...")

try:
    df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\Classification\data.csv', sep=";")
except Exception:
    df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\Classification\data.csv')

df.columns = df.columns.str.strip()
df = df.drop_duplicates()

print("\nDataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Info:")
print(df.info())


target_col = "Target"

if target_col not in df.columns:
    raise ValueError("Target column not found in dataset.")


print("\nTarget Classes:")
print(df[target_col].value_counts())


for col in df.columns:
    if col != target_col:
        df[col] = pd.to_numeric(df[col], errors="coerce")


print("\nApplying NumPy and Pandas operations...")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\nNumeric Columns Count:", len(numeric_cols))
print("\nStatistical Summary:")
print(df[numeric_cols].describe())

df["Total Approved Units"] = (
    df["Curricular units 1st sem (approved)"] +
    df["Curricular units 2nd sem (approved)"]
)

df["Average Grade"] = (
    df["Curricular units 1st sem (grade)"] +
    df["Curricular units 2nd sem (grade)"]
) / 2

df["Total Enrolled Units"] = (
    df["Curricular units 1st sem (enrolled)"] +
    df["Curricular units 2nd sem (enrolled)"]
)

df["Approval Rate"] = np.where(
    df["Total Enrolled Units"] == 0,
    0,
    df["Total Approved Units"] / df["Total Enrolled Units"]
)

df.to_csv(os.path.join(output_dir, "cleaned_dataset.csv"), index=False)


print("\nCreating 5 seaborn maps...")


plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x=target_col,
    order=df[target_col].value_counts().index
)
plt.title("Map 1: Student Target Class Count")
plt.xlabel("Target")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "01_target_count.png"), dpi=300)
plt.close()


plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="Age at enrollment",
    hue=target_col,
    bins=30,
    kde=True
)
plt.title("Map 2: Age Distribution by Target")
plt.xlabel("Age at Enrollment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "02_age_distribution_by_target.png"), dpi=300)
plt.close()


plt.figure(figsize=(9, 5))
sns.boxplot(
    data=df,
    x=target_col,
    y="Admission grade"
)
plt.title("Map 3: Admission Grade by Target")
plt.xlabel("Target")
plt.ylabel("Admission Grade")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "03_admission_grade_by_target.png"), dpi=300)
plt.close()


plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Curricular units 1st sem (approved)",
    y="Curricular units 2nd sem (approved)",
    hue=target_col,
    alpha=0.75
)
plt.title("Map 4: Approved Units 1st Sem vs 2nd Sem")
plt.xlabel("1st Semester Approved Units")
plt.ylabel("2nd Semester Approved Units")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "04_approved_units_scatter.png"), dpi=300)
plt.close()


important_corr_cols = [
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Total Approved Units",
    "Average Grade",
    "Approval Rate",
    "GDP"
]

important_corr_cols = [col for col in important_corr_cols if col in df.columns]

plt.figure(figsize=(11, 7))
sns.heatmap(
    df[important_corr_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Map 5: Important Numeric Features Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "05_correlation_heatmap.png"), dpi=300)
plt.close()


print("5 seaborn maps saved in:", plot_dir)


print("\nStarting Machine Learning Classification...")

X = df.drop(columns=[target_col])
y = df[target_col]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(kernel="rbf", class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}


classification_results = []
trained_models = {}


for model_name, model in classification_models.items():
    print("\n" + "=" * 70)
    print("Training:", model_name)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    classification_results.append({
        "Model": model_name,
        "Accuracy": accuracy
    })

    trained_models[model_name] = pipeline

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )


classification_results_df = pd.DataFrame(classification_results)
classification_results_df = classification_results_df.sort_values(
    by="Accuracy",
    ascending=False
)

classification_results_path = os.path.join(output_dir, "classification_results.csv")
classification_results_df.to_csv(classification_results_path, index=False)

print("\n" + "=" * 70)
print("Classification Accuracy Ranking:")
print(classification_results_df)


plt.figure(figsize=(10, 5))
sns.barplot(
    data=classification_results_df,
    x="Accuracy",
    y="Model"
)
plt.title("Classification Model Accuracy Comparison")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "classification_accuracy_comparison.png"), dpi=300)
plt.close()


best_model_name = classification_results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

best_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, xticks_rotation=45)
plt.title("Best Classification Model Confusion Matrix: " + best_model_name)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_classification_confusion_matrix.png"), dpi=300)
plt.close()


joblib.dump(best_model, os.path.join(model_dir, "best_classification_model.pkl"))
joblib.dump(label_encoder, os.path.join(model_dir, "target_label_encoder.pkl"))

print("\nBest Classification Model:", best_model_name)
print("Best Classification Accuracy:", round(classification_results_df.iloc[0]["Accuracy"] * 100, 2), "%")
print("Best classification model saved in:", os.path.join(model_dir, "best_classification_model.pkl"))


print("\nStarting Machine Learning Clustering...")

X_cluster = df.drop(columns=[target_col])

cluster_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_cluster_scaled = cluster_pipeline.fit_transform(X_cluster)

number_of_clusters = df[target_col].nunique()

kmeans = KMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    n_init=20
)

df["Predicted Cluster"] = kmeans.fit_predict(X_cluster_scaled)


ari_score = adjusted_rand_score(df[target_col], df["Predicted Cluster"])
nmi_score = normalized_mutual_info_score(df[target_col], df["Predicted Cluster"])
sil_score = silhouette_score(X_cluster_scaled, df["Predicted Cluster"])


print("\n" + "=" * 70)
print("Clustering Results")
print("Number of Clusters:", number_of_clusters)
print("Adjusted Rand Index:", round(ari_score, 4))
print("Normalized Mutual Information:", round(nmi_score, 4))
print("Silhouette Score:", round(sil_score, 4))


cluster_summary = pd.crosstab(
    df["Predicted Cluster"],
    df[target_col]
)

print("\nCluster Summary:")
print(cluster_summary)

cluster_summary.to_csv(os.path.join(output_dir, "cluster_summary.csv"))
df.to_csv(os.path.join(output_dir, "dataset_with_predicted_clusters.csv"), index=False)


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster_scaled)

cluster_plot_df = pd.DataFrame({
    "PCA 1": X_pca[:, 0],
    "PCA 2": X_pca[:, 1],
    "Predicted Cluster": df["Predicted Cluster"].astype(str),
    "Actual Target": df[target_col]
})

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=cluster_plot_df,
    x="PCA 1",
    y="PCA 2",
    hue="Predicted Cluster",
    alpha=0.75
)
plt.title("KMeans Clustering Result using PCA")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "kmeans_cluster_pca_plot.png"), dpi=300)
plt.close()


joblib.dump(kmeans, os.path.join(model_dir, "kmeans_clustering_model.pkl"))
joblib.dump(cluster_pipeline, os.path.join(model_dir, "clustering_preprocessing_pipeline.pkl"))

print("\nClustering model saved in:", os.path.join(model_dir, "kmeans_clustering_model.pkl"))
print("Cluster summary saved in:", os.path.join(output_dir, "cluster_summary.csv"))
print("Dataset with clusters saved in:", os.path.join(output_dir, "dataset_with_predicted_clusters.csv"))