# Space Industry Analytics ML + Deep Learning Project
# Uses NumPy, Pandas, Seaborn, 3 ML models, RNN, LSTM, and GRU

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


file_path = r"C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Final Assessment\Space_Industry_Analytics_2010_2024.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    csv_files = [file for file in os.listdir() if file.endswith(".csv")]

    if len(csv_files) > 0:
        df = pd.read_csv(csv_files[0])
        print("Using CSV file:", csv_files[0])
    else:
        raise FileNotFoundError("CSV file not found. Please check your file path.")

df.columns = df.columns.str.strip()
df = df.drop_duplicates()

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

if "Year" in df.columns and "Success_Rate_%" in df.columns and "Company" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Year", y="Success_Rate_%", hue="Company", marker="o")
    plt.title("Success Rate Over Years")
    plt.tight_layout()
    plt.show()

if "Company" in df.columns and "Launches" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Company", y="Launches", estimator="sum")
    plt.title("Total Launches by Company")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if "Revenue_USD_M" in df.columns and "Launches" in df.columns and "Company" in df.columns and "Employees" in df.columns:
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Revenue_USD_M",
        y="Launches",
        hue="Company",
        size="Employees",
        sizes=(50, 300)
    )
    plt.title("Revenue vs Launches")
    plt.tight_layout()
    plt.show()

if "Success_Rate_%" not in df.columns:
    raise ValueError("Column 'Success_Rate_%' not found in dataset.")

df["High_Success"] = (df["Success_Rate_%"] >= 80).astype(int)

print("\nTarget Distribution:")
print(df["High_Success"].value_counts())

X = df.drop(columns=["Success_Rate_%", "High_Success"])
y = df["High_Success"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

try:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", onehot, categorical_cols)
    ]
)

ml_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42
    )
}

class_counts = y.value_counts()

if len(class_counts) < 2:
    raise ValueError("Target column has only one class. Machine learning needs at least two classes.")

stratify_value = y if class_counts.min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=stratify_value
)

print("\n================ MACHINE LEARNING RESULTS ================\n")

best_model_name = None
best_accuracy = 0
best_pipeline = None

for name, model in ml_models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n{name}")
    print("Accuracy:", round(acc * 100, 2), "%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name
        best_pipeline = pipeline

print("\nBest ML Model:", best_model_name)
print("Best ML Accuracy:", round(best_accuracy * 100, 2), "%")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.utils.class_weight import compute_class_weight

    print("\n================ DEEP LEARNING TIME SERIES ================\n")

    tf.random.set_seed(42)
    np.random.seed(42)

    required_columns = [
        "Year",
        "Launches",
        "Successful",
        "Failed",
        "Revenue_USD_M",
        "Budget_Funding_USD_M",
        "Employees",
        "Rockets",
        "Success_Rate_%",
        "Company",
        "High_Success"
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]

    if len(missing_cols) > 0:
        print("Deep Learning skipped because these columns are missing:")
        print(missing_cols)
    else:
        LOOKBACK = 3

        sequence_features = [
            "Year",
            "Launches",
            "Successful",
            "Failed",
            "Revenue_USD_M",
            "Budget_Funding_USD_M",
            "Employees",
            "Rockets",
            "Success_Rate_%"
        ]

        ts_df = df.sort_values(["Company", "Year"]).copy()

        scaler = MinMaxScaler()
        ts_df[sequence_features] = scaler.fit_transform(ts_df[sequence_features])

        X_seq = []
        y_seq = []

        for company, group in ts_df.groupby("Company"):
            group = group.sort_values("Year")

            for i in range(len(group) - LOOKBACK):
                X_seq.append(group[sequence_features].iloc[i:i + LOOKBACK].values)
                y_seq.append(group["High_Success"].iloc[i + LOOKBACK])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.int32)

        print("Sequence Data Shape:", X_seq.shape)
        print("Sequence Target Shape:", y_seq.shape)

        if len(X_seq) < 5:
            print("Deep Learning skipped because sequence data is too small.")
        elif len(np.unique(y_seq)) < 2:
            print("Deep Learning skipped because target has only one class.")
        else:
            seq_class_counts = pd.Series(y_seq).value_counts()
            seq_stratify = y_seq if seq_class_counts.min() >= 2 else None

            X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
                X_seq,
                y_seq,
                test_size=0.30,
                random_state=42,
                stratify=seq_stratify
            )

            classes = np.unique(y_seq_train)

            class_weights_array = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=y_seq_train
            )

            class_weights = {
                int(cls): float(weight)
                for cls, weight in zip(classes, class_weights_array)
            }

            def build_rnn_model(model_type, input_shape):
                model = Sequential()

                if model_type == "RNN":
                    model.add(SimpleRNN(64, activation="tanh", input_shape=input_shape))
                elif model_type == "LSTM":
                    model.add(LSTM(64, activation="tanh", input_shape=input_shape))
                elif model_type == "GRU":
                    model.add(GRU(64, activation="tanh", input_shape=input_shape))

                model.add(Dropout(0.25))
                model.add(Dense(32, activation="relu"))
                model.add(Dropout(0.20))
                model.add(Dense(1, activation="sigmoid"))

                model.compile(
                    optimizer="adam",
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )

                return model

            deep_models = ["RNN", "LSTM", "GRU"]

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True
            )

            input_shape = (X_seq_train.shape[1], X_seq_train.shape[2])

            for model_type in deep_models:
                print(f"\n{model_type} Model")

                model = build_rnn_model(model_type, input_shape)

                history = model.fit(
                    X_seq_train,
                    y_seq_train,
                    epochs=120,
                    batch_size=4,
                    validation_split=0.20,
                    callbacks=[early_stop],
                    class_weight=class_weights,
                    verbose=0
                )

                loss, accuracy = model.evaluate(X_seq_test, y_seq_test, verbose=0)

                print("Test Accuracy:", round(accuracy * 100, 2), "%")
                print("Test Loss:", round(loss, 4))

                y_prob = model.predict(X_seq_test, verbose=0)
                y_pred_dl = (y_prob >= 0.5).astype(int)

                print("Confusion Matrix:")
                print(confusion_matrix(y_seq_test, y_pred_dl))

                print("Classification Report:")
                print(classification_report(y_seq_test, y_pred_dl, zero_division=0))

                plt.figure(figsize=(8, 5))
                plt.plot(history.history["accuracy"], label="Training Accuracy")
                plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
                plt.title(f"{model_type} Accuracy Curve")
                plt.xlabel("Epochs")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.tight_layout()
                plt.show()

except ImportError:
    print("\nTensorFlow is not installed.")
    print("Install TensorFlow by running this command in terminal:")
    print("pip install tensorflow")
    print("If you are using Python 3.12 and TensorFlow does not install, use Python 3.10 or 3.11.")

except Exception as e:
    print("\nDeep Learning Error:")
    print(e)

print("\nFinal assessment question 2 part 1 dataset using deep learning end to end completed.")

# This code cleans and studies space industry data, then shows charts like heatmap, success rate, launches by company, and revenue vs launches.
# It uses machine learning models like Logistic Regression, Random Forest, and Gradient Boosting to predict high success rates.
# It also uses deep learning models like RNN, LSTM, and GRU to learn from yearly company data and compare results.