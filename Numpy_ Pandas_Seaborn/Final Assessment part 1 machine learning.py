import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("=" * 80)
print("AI DATA SCIENCE PROJECT")
print("=" * 80)

# Load Dataset

csv_file =(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Final Assessment\data.csv')

df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Final Assessment\data.csv')

print("\nDataset Loaded Successfully.")

# Basic Information

print("\nFirst Five Rows")
print(df.head())

print("\nLast Five Rows")
print(df.tail())

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nInformation")
df.info()

print("\nStatistical Summary")
print(df.describe(include="all"))

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

print("\nUnique Values in Each Column")

for col in df.columns:
    print(f"{col} : {df[col].nunique()}")

memory = df.memory_usage(deep=True).sum() / 1024**2

print("\nMemory Usage")
print(f"{memory:.2f} MB")

numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumerical Columns")
print(numerical_columns)

print("\nCategorical Columns")
print(categorical_columns)

print("\nCategorical Value Counts")

for col in categorical_columns:
    print("\n", "=" * 60)
    print(col)
    print(df[col].value_counts())

if len(numerical_columns) > 0:

    print("\nNUMPY ANALYSIS")

    for col in numerical_columns:

        array = df[col].dropna().to_numpy()

        if len(array) == 0:
            continue

        print("\nColumn :", col)

        print("Mean :", np.mean(array))

        print("Median :", np.median(array))

        try:
            values, counts = np.unique(array, return_counts=True)
            mode = values[np.argmax(counts)]
            print("Mode :", mode)
        except:
            print("Mode : Not Available")

        print("Minimum :", np.min(array))
        print("Maximum :", np.max(array))
        print("Sum :", np.sum(array))
        print("Variance :", np.var(array))
        print("Standard Deviation :", np.std(array))

        print("25 Percentile :", np.percentile(array, 25))
        print("50 Percentile :", np.percentile(array, 50))
        print("75 Percentile :", np.percentile(array, 75))


print("\nHandling Missing Values")

for col in numerical_columns:

    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_columns:

    if df[col].mode().empty:
        df[col].fillna("Unknown", inplace=True)
    else:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing Values Remaining")
print(df.isnull().sum())

before = df.shape[0]

df.drop_duplicates(inplace=True)

after = df.shape[0]

print("\nDuplicates Removed :", before - after)

for col in df.columns:

    try:

        converted = pd.to_datetime(df[col], errors="coerce")

        if converted.notnull().sum() > len(df) * 0.80:

            df[col] = converted.map(pd.Timestamp.toordinal)

            print(f"{col} converted to numeric date.")

    except:

        pass
     

        pass

constant_columns = []

for col in df.columns:

    if df[col].nunique() <= 1:

        constant_columns.append(col)

if len(constant_columns) > 0:

    print("\nRemoving Constant Columns")

    print(constant_columns)

    df.drop(columns=constant_columns, inplace=True)

print("\nRemoving Outliers")

for col in numerical_columns:

    if col not in df.columns:
        continue

    Q1 = df[col].quantile(0.25)

    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

print("Outliers Removed.")

if len(df.select_dtypes(include=np.number).columns) > 1:

    corr = df.corr(numeric_only=True)

    print("\nCorrelation Matrix")

    print(corr)

df.to_csv("cleaned_data.csv", index=False)

print("\nCleaned Dataset Saved Successfully.")

print("\nCurrent Shape")

print(df.shape)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

print("\n" + "=" * 80)
print("PART 2 : FEATURE ENGINEERING & VISUALIZATION")
print("=" * 80)

target_column = df.columns[-1]

print("\nTarget Column Detected :", target_column)

label_encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:

    if df[col].nunique() <= 20:

        df[col] = label_encoder.fit_transform(df[col].astype(str))

        print(f"{col} Label Encoded")


remaining_object = df.select_dtypes(include="object").columns.tolist()

if len(remaining_object) > 0:

    df = pd.get_dummies(
        df,
        columns=remaining_object,
        drop_first=True
    )

    print("\nOne Hot Encoding Applied")

print("\nDataset Shape After Encoding")

print(df.shape)

X = df.drop(columns=[target_column])

y = df[target_column]

numerical_features = X.select_dtypes(include=np.number).columns

standard_scaler = StandardScaler()

X_standard = X.copy()

X_standard[numerical_features] = standard_scaler.fit_transform(
    X_standard[numerical_features]
)

print("\nStandard Scaling Completed")

minmax_scaler = MinMaxScaler()

X_minmax = X.copy()

X_minmax[numerical_features] = minmax_scaler.fit_transform(
    X_minmax[numerical_features]
)

print("MinMax Scaling Completed")

# Pandas Operations

print("\nSorting Dataset")

sorted_df = df.sort_values(target_column)

print(sorted_df.head())

print("\nFiltering Numerical Rows")

if len(numerical_features) > 0:

    first_num = numerical_features[0]

    filtered = df[df[first_num] > df[first_num].mean()]

    print(filtered.head())

print("\nGroupBy")

if len(df.columns) >= 2:

    first_col = df.columns[0]

    second_col = df.columns[1]

    try:

        grouped = df.groupby(first_col)[second_col].count()

        print(grouped)

    except:

        print("GroupBy Not Possible")

print("\nAggregation")

try:

    aggregation = df.agg(["mean", "median", "std"], numeric_only=True)

    print(aggregation)

except:

    pass

print("\nPivot Table")

try:

    if len(df.columns) >= 3:

        pivot = pd.pivot_table(
            df,
            values=df.select_dtypes(include=np.number).columns[0],
            index=df.columns[0],
            aggfunc=np.mean
        )

        print(pivot.head())

except:

    print("Pivot Table Not Possible")

print("\nCrosstab")

try:

    if len(df.columns) >= 2:

        cross = pd.crosstab(df[df.columns[0]], df[target_column])

        print(cross.head())

except:

    print("Crosstab Not Possible")

print("\nApply Function")

for col in numerical_features:

    df[col] = df[col].apply(lambda x: round(x, 2))

print("Apply Function Completed")

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()

if len(numerical_features) > 0:

    plt.figure(figsize=(8,5))

    sns.histplot(
        df[numerical_features[0]],
        kde=True
    )

    plt.title("Histogram")

    plt.tight_layout()

    plt.show()

if len(numerical_features) >= 2:

    plt.figure(figsize=(8,5))

    sns.scatterplot(
        x=df[numerical_features[0]],
        y=df[numerical_features[1]]
    )

    plt.title("Scatter Plot")

    plt.tight_layout()

    plt.show()

if len(numerical_features) > 0:

    plt.figure(figsize=(8,5))

    sns.boxplot(
        x=df[numerical_features[0]]
    )

    plt.title("Box Plot")

    plt.tight_layout()

    plt.show()

if target_column in df.columns:

    plt.figure(figsize=(8,5))

    sns.countplot(
        x=df[target_column]
    )

    plt.title("Count Plot")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()

df.to_csv(
    "encoded_dataset.csv",
    index=False
)

print("\nEncoded Dataset Saved Successfully.")

print("\nCurrent Dataset Shape")

print(df.shape)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.svm import SVC, SVR

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

print("\n" + "=" * 80)
print("PART 3 : MACHINE LEARNING")
print("=" * 80)

problem_type = "Classification" if y.nunique() <= 20 else "Regression"

print("\nProblem Type :", problem_type)

# Make sure only numeric features are used
X_standard = X_standard.select_dtypes(include=["number"])

# Remove missing values if any
X_standard = X_standard.fillna(0)

if problem_type == "Classification":

    X_train, X_test, y_train, y_test = train_test_split(
    X_standard,
    y,
    test_size=0.20,
    random_state=42
)

else:

    X_train, X_test, y_train, y_test = train_test_split(
        X_standard,
        y,
        test_size=0.20,
        random_state=42
    )

print("\nTraining Samples :", X_train.shape)
print("Testing Samples :", X_test.shape)

results = {}
trained_models = {}

if problem_type == "Classification":

    models = {

        "Logistic Regression":
        LogisticRegression(max_iter=1000),

        "Decision Tree":
        DecisionTreeClassifier(max_depth=10, random_state=42),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

        "Support Vector Machine":
        SVC(kernel="rbf", C=10, gamma="scale"),

        "K-Nearest Neighbors":
        KNeighborsClassifier(n_neighbors=5)

    }

    print("\nTraining Classification Models...\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        accuracy = accuracy_score(y_test, prediction)

        results[name] = accuracy

        trained_models[name] = model

        print(f"{name} Accuracy : {accuracy*100:.2f}%")

else:

    models = {

        "Linear Regression":
        LinearRegression(),

        "Decision Tree":
        DecisionTreeRegressor(random_state=42),

        "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),

        "Support Vector Machine":
        SVR(),

        "K-Nearest Neighbors":
        KNeighborsRegressor()

    }

    print("\nTraining Regression Models...\n")

    for name, model in models.items():

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        r2 = r2_score(y_test, prediction)

        mae = mean_absolute_error(y_test, prediction)

        rmse = np.sqrt(mean_squared_error(y_test, prediction))

        results[name] = r2

        trained_models[name] = model

        print(name)
        print("R2 Score :", round(r2,4))
        print("MAE :", round(mae,4))
        print("RMSE :", round(rmse,4))
        print("-"*50)

comparison = pd.DataFrame({

    "Model": list(results.keys()),
    "Score": list(results.values())

})

comparison = comparison.sort_values(
    by="Score",
    ascending=False
).reset_index(drop=True)

print("\n")
print("="*80)
print("MODEL COMPARISON")
print("="*80)

print(comparison)

best_model_name = comparison.loc[0, "Model"]
best_score = comparison.loc[0, "Score"]

print("\nBest Model :", best_model_name)

if problem_type == "Classification":

    print(f"Best Accuracy : {best_score*100:.2f}%")

else:

    print(f"Best R2 Score : {best_score:.4f}")

print("\nQuestion 1 Part 1 of final assessment is completed.")
print("\nI use numpy,pandas,seaborn and machine learning for solving problem.")

