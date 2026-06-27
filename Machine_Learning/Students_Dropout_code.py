import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

df = pd.read_csv(
    r"C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\Students_Dropout.csv",
    delimiter=";"
)

print("df - DataFrame:\n", df)

print("\n===================================")
print("df.info()")
print(df.info())

print("\n===================================")
print("df.describe()")
print(df.describe())

print("\n===================================")
print("df.head()")
print(df.head())

print("\n===================================")
print("df.tail()")
print(df.tail())

label_encoder = LabelEncoder()

for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

print("\n===================================")
print("Encoded DataFrame")
print(df.head())

X = df.drop("Target", axis=1).copy()
y = df["Target"].copy()

print("\n===================================")
print("X:")
print(X)

print("\n===================================")
print("y:")
print(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n===================================")
print("First Row after Scaling:")
print(X_scaled[0])

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled,
    y,
    train_size=0.7,
    random_state=25,
    stratify=y
)

print(f"Train size: {round(len(X_train_scaled) / len(X) * 100)}%")
print(f"Test size: {round(len(X_test_scaled) / len(X) * 100)}%")

param_grid = {
    'C': [1, 10, 50, 100, 200],
    'gamma': [1, 0.1, 0.01, 0.001, 'scale'],
    'kernel': ['rbf']
}

svm = SVC()

grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print("\n===================================")
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(f"{grid_search.best_score_ * 100:.2f}%")

best_svm = grid_search.best_estimator_

svm_preds = best_svm.predict(X_test_scaled)

print("\n===================================")
print("SVM Results:\n")
print(classification_report(y_test, svm_preds))

accuracy = accuracy_score(y_test, svm_preds)

print("Accuracy Score:")
print(f"{accuracy * 100:.2f}%")

cm = confusion_matrix(y_test, svm_preds)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    linewidths=0.5,
    linecolor='black'
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

accuracy_df = pd.DataFrame({
    "Model": ["SVM"],
    "Accuracy": [accuracy * 100]
})

plt.figure(figsize=(4, 2))

sns.heatmap(
    accuracy_df[["Accuracy"]],
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    cbar=True,
    yticklabels=accuracy_df["Model"]
)

plt.title("SVM Accuracy Heatmap")
plt.tight_layout()
plt.show()

print("\n===================================")
print("Final SVM Model:")
print(best_svm)

print("\nFinal Test Accuracy:")
print(f"{accuracy * 100:.2f}%")