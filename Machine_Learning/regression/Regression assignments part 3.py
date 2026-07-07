import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

DATA_FILE =r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\regression\housing.csv' 

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("housing.csv not found. Put the CSV file in the same folder as this Python file.")

df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\regression\housing.csv')
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nDataset Loaded Successfully")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

numpy_data = df.select_dtypes(include=[np.number]).to_numpy()

print("\nNumPy Array Shape:")
print(numpy_data.shape)

print("\nNumPy Mean Values:")
print(np.round(np.nanmean(numpy_data, axis=0), 2))

print("\nNumPy Standard Deviation:")
print(np.round(np.nanstd(numpy_data, axis=0), 2))

print("\nPandas Description:")
print(df.describe())

print("\nPandas Correlation:")
print(df.corr(numeric_only=True))

os.makedirs("outputs", exist_ok=True)

df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

df["rooms_per_household"] = df["rooms_per_household"].replace([np.inf, -np.inf], np.nan)
df["bedrooms_per_room"] = df["bedrooms_per_room"].replace([np.inf, -np.inf], np.nan)
df["population_per_household"] = df["population_per_household"].replace([np.inf, -np.inf], np.nan)

df["income_category"] = pd.cut(
    df["median_income"],
    bins=[0, 1.5, 3, 4.5, 6, np.inf],
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
)

sample_df = df.sample(n=min(3000, len(df)), random_state=42)

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("1. Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/1_correlation_heatmap.png", dpi=300)
plt.close()

sns.pairplot(
    sample_df[[
        "median_income",
        "housing_median_age",
        "rooms_per_household",
        "population_per_household",
        "median_house_value"
    ]],
    diag_kind="kde"
)
plt.savefig("outputs/2_pairplot.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(df["median_house_value"], kde=True)
plt.title("3. Median House Value Distribution")
plt.xlabel("Median House Value")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/3_house_value_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(df["median_income"], kde=True)
plt.title("4. Median Income Distribution")
plt.xlabel("Median Income")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/4_income_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(x="ocean_proximity", data=df)
plt.title("5. Ocean Proximity Count")
plt.xlabel("Ocean Proximity")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/5_ocean_proximity_count.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
sns.boxplot(x="ocean_proximity", y="median_house_value", data=df)
plt.title("6. Ocean Proximity vs House Value")
plt.xlabel("Ocean Proximity")
plt.ylabel("Median House Value")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/6_ocean_boxplot.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
sns.violinplot(x="ocean_proximity", y="median_house_value", data=df)
plt.title("7. Ocean Proximity House Value Violin Plot")
plt.xlabel("Ocean Proximity")
plt.ylabel("Median House Value")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/7_ocean_violinplot.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.regplot(x="median_income", y="median_house_value", data=sample_df, scatter_kws={"alpha": 0.4})
plt.title("8. Median Income vs House Value")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.tight_layout()
plt.savefig("outputs/8_income_vs_house_value.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.regplot(x="housing_median_age", y="median_house_value", data=sample_df, scatter_kws={"alpha": 0.4})
plt.title("9. Housing Age vs House Value")
plt.xlabel("Housing Median Age")
plt.ylabel("Median House Value")
plt.tight_layout()
plt.savefig("outputs/9_age_vs_house_value.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 6))
sns.scatterplot(
    x="longitude",
    y="latitude",
    hue="median_house_value",
    size="population",
    data=sample_df,
    alpha=0.6
)
plt.title("10. Location Map by House Value")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("outputs/10_location_house_value_map.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 6))
sns.scatterplot(
    x="longitude",
    y="latitude",
    hue="ocean_proximity",
    data=sample_df,
    alpha=0.7
)
plt.title("11. Location Map by Ocean Proximity")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("outputs/11_location_ocean_map.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.regplot(x="rooms_per_household", y="median_house_value", data=sample_df, scatter_kws={"alpha": 0.4})
plt.title("12. Rooms per Household vs House Value")
plt.xlabel("Rooms per Household")
plt.ylabel("Median House Value")
plt.xlim(0, 10)
plt.tight_layout()
plt.savefig("outputs/12_rooms_per_household.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.regplot(x="bedrooms_per_room", y="median_house_value", data=sample_df, scatter_kws={"alpha": 0.4})
plt.title("13. Bedrooms per Room vs House Value")
plt.xlabel("Bedrooms per Room")
plt.ylabel("Median House Value")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("outputs/13_bedrooms_per_room.png", dpi=300)
plt.close()

X = df.drop(["median_house_value", "income_category"], axis=1)
y = df["median_house_value"]

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

numeric_transformer_scaled = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_transformer_tree = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", encoder)
])

preprocessor_scaled = ColumnTransformer([
    ("num", numeric_transformer_scaled, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

preprocessor_tree = ColumnTransformer([
    ("num", numeric_transformer_tree, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

models = {
    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor_scaled),
        ("model", LinearRegression())
    ]),

    "Ridge Regression": Pipeline([
        ("preprocessor", preprocessor_scaled),
        ("model", Ridge(alpha=10))
    ]),

    "Lasso Regression": Pipeline([
        ("preprocessor", preprocessor_scaled),
        ("model", Lasso(alpha=0.001, max_iter=10000))
    ]),

    "Decision Tree Regression": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", DecisionTreeRegressor(
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ))
    ]),

    "Random Forest Regression": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", RandomForestRegressor(
            n_estimators=250,
            max_depth=22,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Gradient Boosting Regression": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", GradientBoostingRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            random_state=42
        ))
    ]),

    "Extra Trees Regression": Pipeline([
        ("preprocessor", preprocessor_tree),
        ("model", ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

results = []
predictions = pd.DataFrame()
predictions["Actual House Value"] = y_test.values

trained_models = {}

for model_name, model in models.items():
    final_model = TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1
    )

    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({
        "Model": model_name,
        "R2 Score": r2,
        "Accuracy Percentage": accuracy,
        "MAE": mae,
        "RMSE": rmse
    })

    predictions[model_name + " Prediction"] = y_pred
    trained_models[model_name] = final_model

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2 Score", ascending=False)

print("\nModel Results:")
print(results_df)

print("\nPredictions:")
print(predictions.head())

results_df.to_csv("outputs/model_results.csv", index=False)
predictions.to_csv("outputs/predictions.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy Percentage", y="Model", data=results_df)
plt.title("14. Regression Model Accuracy Comparison")
plt.xlabel("Accuracy Percentage using R2 Score")
plt.ylabel("Regression Model")
plt.tight_layout()
plt.savefig("outputs/14_model_accuracy_comparison.png", dpi=300)
plt.close()

best_model_name = results_df.iloc[0]["Model"]
best_prediction_column = best_model_name + " Prediction"

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=predictions["Actual House Value"],
    y=predictions[best_prediction_column],
    alpha=0.6
)
plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title(f"15. Actual vs Predicted - {best_model_name}")
plt.tight_layout()
plt.savefig("outputs/15_actual_vs_predicted.png", dpi=300)
plt.close()

print("\nBest Model:")
print(best_model_name)

print("\nBest Model Accuracy using R2 Percentage:")
print(round(results_df.iloc[0]["Accuracy Percentage"], 2), "%")

print("\nBest Model R2 Score:")
print(round(results_df.iloc[0]["R2 Score"], 4))

print("\nBest Model MAE:")
print(round(results_df.iloc[0]["MAE"], 2))

print("\nBest Model RMSE:")
print(round(results_df.iloc[0]["RMSE"], 2))