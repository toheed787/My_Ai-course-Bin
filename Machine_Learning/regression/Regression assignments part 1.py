import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_FILE = r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\regression\advertising.csv'

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("advertising.csv not found. Put it in the same folder as this Python file.")

df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\regression\advertising.csv')
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nDataset Loaded Successfully")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

numpy_data = df.to_numpy()

print("\nNumPy Array Shape:")
print(numpy_data.shape)

print("\nMean Values using NumPy:")
print(np.round(np.mean(numpy_data, axis=0), 2))

print("\nPandas Description:")
print(df.describe())

print("\nCorrelation using Pandas:")
print(df.corr(numeric_only=True))

os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(8, 5))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=300)
plt.show()

pair_plot = sns.pairplot(df, diag_kind="kde")
pair_plot.fig.suptitle("Pairplot of Advertising Dataset", y=1.02)
pair_plot.savefig("outputs/seaborn_pairplot.png", dpi=300)
plt.show()

X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model_1 = Pipeline([
    ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge_regression", Ridge(alpha=0.1))
])

model_2 = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=2,
    random_state=42
)

models = {
    "Polynomial Ridge Regression": model_1,
    "Gradient Boosting Regression": model_2
}

results = []
predictions = pd.DataFrame()
predictions["Actual Sales"] = y_test.values

for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    accuracy_percent = r2 * 100
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_score = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    results.append({
        "Model": model_name,
        "R2 Score": r2,
        "Accuracy Percentage": accuracy_percent,
        "MAE": mae,
        "RMSE": rmse,
        "Cross Validation R2": cv_score
    })

    predictions[model_name + " Prediction"] = y_pred

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2 Score", ascending=False)

print("\nModel Results:")
print(results_df)

print("\nPredictions:")
print(predictions.head())

results_df.to_csv("outputs/model_results.csv", index=False)
predictions.to_csv("outputs/predictions.csv", index=False)

best_model_name = results_df.iloc[0]["Model"]
best_prediction_column = best_model_name + " Prediction"

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=predictions["Actual Sales"],
    y=predictions[best_prediction_column]
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales - {best_model_name}")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png", dpi=300)
plt.show()

print("\nBest Model:")
print(best_model_name)

print("\nBest Model Accuracy using R2 Percentage:")
print(round(results_df.iloc[0]["Accuracy Percentage"], 2), "%")