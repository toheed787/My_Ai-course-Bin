import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_FILE = r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\regression\Boston.csv'

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\regression\Boston.csv')

df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\regression\Boston.csv')
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nDataset Loaded Successfully")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

numpy_data = df.to_numpy()

print("\nNumPy Array Shape:")
print(numpy_data.shape)

print("\nNumPy Mean Values:")
print(np.round(np.mean(numpy_data, axis=0), 2))

print("\nNumPy Standard Deviation:")
print(np.round(np.std(numpy_data, axis=0), 2))

print("\nPandas Description:")
print(df.describe())

print("\nPandas Correlation:")
print(df.corr(numeric_only=True))

os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/1_correlation_heatmap.png", dpi=300)
plt.show()

sns.pairplot(df[["rm", "lstat", "ptratio", "medv"]], diag_kind="kde")
plt.savefig("outputs/2_pairplot.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["medv"], kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("MEDV")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/3_price_distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.regplot(x="rm", y="medv", data=df)
plt.title("Rooms vs House Price")
plt.xlabel("Average Number of Rooms")
plt.ylabel("MEDV")
plt.tight_layout()
plt.savefig("outputs/4_rooms_vs_price.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
sns.regplot(x="lstat", y="medv", data=df)
plt.title("Lower Status Population vs House Price")
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.tight_layout()
plt.savefig("outputs/5_lstat_vs_price.png", dpi=300)
plt.show()

df_model = df.copy()

if "rm" in df_model.columns:
    df_model["rm_squared"] = df_model["rm"] ** 2

if "lstat" in df_model.columns:
    df_model["lstat_squared"] = df_model["lstat"] ** 2

if "rm" in df_model.columns and "lstat" in df_model.columns:
    df_model["rm_lstat"] = df_model["rm"] * df_model["lstat"]

if "tax" in df_model.columns and "rad" in df_model.columns:
    df_model["tax_rad"] = df_model["tax"] * df_model["rad"]

X = df_model.drop("medv", axis=1)
y = df_model["medv"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model_1 = Pipeline([
    ("polynomial_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge_regression", Ridge(alpha=10))
])

model_2 = RandomForestRegressor(
    n_estimators=500,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model_3 = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.9,
    random_state=42
)

model_4 = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

models = {
    "Polynomial Ridge Regression": model_1,
    "Random Forest Regression": model_2,
    "Gradient Boosting Regression": model_3,
    "Extra Trees Regression": model_4
}

results = []
predictions = pd.DataFrame()
predictions["Actual Price"] = y_test.values

for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()

    results.append({
        "Model": model_name,
        "R2 Score": r2,
        "Accuracy Percentage": accuracy,
        "MAE": mae,
        "RMSE": rmse,
        "Cross Validation R2": cv_r2
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
    x=predictions["Actual Price"],
    y=predictions[best_prediction_column]
)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted Price - {best_model_name}")
plt.tight_layout()
plt.savefig("outputs/6_actual_vs_predicted.png", dpi=300)
plt.show()

plt.figure(figsize=(9, 5))
sns.barplot(
    x="Accuracy Percentage",
    y="Model",
    data=results_df
)
plt.title("Model Accuracy Comparison")
plt.xlabel("Accuracy Percentage using R2 Score")
plt.ylabel("Regression Models")
plt.tight_layout()
plt.savefig("outputs/7_model_accuracy_comparison.png", dpi=300)
plt.show()

print("\nBest Model:")
print(best_model_name)

print("\nBest Model Accuracy:")
print(round(results_df.iloc[0]["Accuracy Percentage"], 2), "%")