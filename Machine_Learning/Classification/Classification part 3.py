import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.cluster import MiniBatchKMeans

import joblib


file_path = r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\Classification\pricerunner_aggregate.csv'

if not os.path.exists(file_path):
    raise FileNotFoundError("CSV file not found. Put pricerunner_aggregate.csv in the same folder as main.py")


output_dir = "pricerunner_outputs"
plot_dir = os.path.join(output_dir, "seaborn_maps")
model_dir = os.path.join(output_dir, "models")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

sns.set_theme(style="whitegrid")


print("\nLoading dataset...")
df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\Classification\pricerunner_aggregate.csv')

df.columns = df.columns.str.strip()
df = df.drop_duplicates()

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


required_columns = [
    "Product ID",
    "Product Title",
    "Merchant ID",
    "Cluster ID",
    "Cluster Label",
    "Category ID",
    "Category Label"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError("Missing columns in dataset: " + str(missing_columns))


df["Product Title"] = df["Product Title"].astype(str)
df["Cluster Label"] = df["Cluster Label"].astype(str)
df["Category Label"] = df["Category Label"].astype(str)

df["Title Length"] = df["Product Title"].str.len()
df["Word Count"] = df["Product Title"].str.split().str.len()


print("\nCategory Counts:")
print(df["Category Label"].value_counts())


print("\nCreating 5 seaborn maps...")


plt.figure(figsize=(10, 6))
sns.countplot(
    data=df,
    y="Category Label",
    order=df["Category Label"].value_counts().index
)
plt.title("Map 1: Product Count by Category")
plt.xlabel("Count")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "01_category_count.png"), dpi=300)
plt.close()


top_merchants = df["Merchant ID"].value_counts().head(15).reset_index()
top_merchants.columns = ["Merchant ID", "Count"]
top_merchants["Merchant ID"] = top_merchants["Merchant ID"].astype(str)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_merchants,
    x="Count",
    y="Merchant ID"
)
plt.title("Map 2: Top 15 Merchants by Product Count")
plt.xlabel("Product Count")
plt.ylabel("Merchant ID")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "02_top_merchants.png"), dpi=300)
plt.close()


plt.figure(figsize=(10, 6))
sns.histplot(
    data=df,
    x="Title Length",
    bins=40,
    kde=True
)
plt.title("Map 3: Product Title Length Distribution")
plt.xlabel("Title Length")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "03_title_length_distribution.png"), dpi=300)
plt.close()


plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df,
    x="Title Length",
    y="Category Label"
)
plt.title("Map 4: Title Length by Category")
plt.xlabel("Title Length")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "04_title_length_by_category.png"), dpi=300)
plt.close()


numeric_df = df[["Product ID", "Merchant ID", "Cluster ID", "Category ID", "Title Length", "Word Count"]]

plt.figure(figsize=(9, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Map 5: Numeric Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "05_correlation_heatmap.png"), dpi=300)
plt.close()


print("\n5 seaborn maps saved in:", plot_dir)


print("\nStarting Machine Learning Classification...")

X = df["Product Title"]
y = df["Category Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


classification_models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Linear SVM": LinearSVC()
}


classification_results = []
trained_classification_models = {}


for model_name, model in classification_models.items():
    print("\n" + "=" * 70)
    print("Training:", model_name)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    classification_results.append({
        "Model": model_name,
        "Accuracy": accuracy
    })

    trained_classification_models[model_name] = pipeline

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


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


plt.figure(figsize=(9, 5))
sns.barplot(
    data=classification_results_df,
    x="Accuracy",
    y="Model"
)
plt.title("Classification Model Accuracy")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "classification_accuracy_comparison.png"), dpi=300)
plt.close()


best_classification_model_name = classification_results_df.iloc[0]["Model"]
best_classification_model = trained_classification_models[best_classification_model_name]

best_y_pred = best_classification_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_y_pred,
    labels=best_classification_model.classes_
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=best_classification_model.classes_
)

fig, ax = plt.subplots(figsize=(11, 8))
disp.plot(ax=ax, xticks_rotation=45)
plt.title("Best Classification Model Confusion Matrix: " + best_classification_model_name)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_classification_confusion_matrix.png"), dpi=300)
plt.close()


joblib.dump(
    best_classification_model,
    os.path.join(model_dir, "best_classification_model.pkl")
)

print("\nBest Classification Model:", best_classification_model_name)
print("Best Classification Accuracy:", round(classification_results_df.iloc[0]["Accuracy"] * 100, 2), "%")
print("Best classification model saved in:", os.path.join(model_dir, "best_classification_model.pkl"))


print("\nStarting Machine Learning Clustering...")

number_of_clusters = df["Category Label"].nunique()

tfidf_clustering = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_cluster = tfidf_clustering.fit_transform(df["Product Title"])

kmeans = MiniBatchKMeans(
    n_clusters=number_of_clusters,
    random_state=42,
    batch_size=1024,
    n_init=10
)

df["Predicted Cluster"] = kmeans.fit_predict(X_cluster)


ari_score = adjusted_rand_score(df["Category Label"], df["Predicted Cluster"])
nmi_score = normalized_mutual_info_score(df["Category Label"], df["Predicted Cluster"])

sample_size = min(5000, X_cluster.shape[0])
sample_index = np.random.RandomState(42).choice(
    X_cluster.shape[0],
    size=sample_size,
    replace=False
)

sil_score = silhouette_score(
    X_cluster[sample_index],
    df["Predicted Cluster"].iloc[sample_index]
)


print("\n" + "=" * 70)
print("Clustering Results")
print("Number of Clusters:", number_of_clusters)
print("Adjusted Rand Index:", round(ari_score, 4))
print("Normalized Mutual Information:", round(nmi_score, 4))
print("Silhouette Score:", round(sil_score, 4))


cluster_summary = pd.crosstab(
    df["Predicted Cluster"],
    df["Category Label"]
)

cluster_summary_path = os.path.join(output_dir, "cluster_summary.csv")
cluster_summary.to_csv(cluster_summary_path)

df_output_path = os.path.join(output_dir, "dataset_with_predicted_clusters.csv")
df.to_csv(df_output_path, index=False)


joblib.dump(
    kmeans,
    os.path.join(model_dir, "kmeans_clustering_model.pkl")
)

joblib.dump(
    tfidf_clustering,
    os.path.join(model_dir, "tfidf_clustering_vectorizer.pkl")
)


print("\nCluster Summary:")
print(cluster_summary)

print("\nClustering model saved in:", os.path.join(model_dir, "kmeans_clustering_model.pkl"))
print("TF-IDF vectorizer saved in:", os.path.join(model_dir, "tfidf_clustering_vectorizer.pkl"))
print("Cluster summary saved in:", cluster_summary_path)
print("Dataset with predicted clusters saved in:", df_output_path)