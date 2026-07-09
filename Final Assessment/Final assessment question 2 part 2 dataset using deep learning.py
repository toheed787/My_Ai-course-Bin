# SpaceX Starlink Project: NumPy, Pandas, Seaborn, Machine Learning

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
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier

try:
    import joblib
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False


file_path = r"C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Final Assessment\spacex_starlink.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("CSV file not found. Put spacex_starlink.csv in the same folder as this Python file.")


output_dir = "spacex_starlink_outputs"
plot_dir = os.path.join(output_dir, "seaborn_maps")
model_dir = os.path.join(output_dir, "saved_models")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

sns.set_theme(style="whitegrid")


print("\nLoading dataset...")
df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Final Assessment\spacex_starlink.csv')

df.columns = df.columns.str.strip()
df = df.drop_duplicates()

numeric_cols = ["spacetrack_id", "launch_date", "longitude", "latitude", "height_km", "velocity_kms"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Info:")
print(df.info())


plot_df = df.copy()

if "version" in plot_df.columns:
    plot_df["version_plot"] = plot_df["version"].fillna("Unknown")
else:
    plot_df["version_plot"] = "Unknown"

available_numeric = [
    col for col in ["longitude", "latitude", "height_km", "velocity_kms"]
    if col in plot_df.columns
]


print("\nCreating 10 seaborn maps/plots...")


plt.figure(figsize=(10, 5))
sns.heatmap(plot_df.isnull(), cbar=False)
plt.title("Map 1: Missing Values Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "01_missing_values_heatmap.png"), dpi=300)
plt.close()


plt.figure(figsize=(8, 5))
sns.countplot(
    data=plot_df,
    x="version_plot",
    hue="version_plot",
    legend=False,
    order=plot_df["version_plot"].value_counts().index
)
plt.title("Map 2: Satellite Version Count")
plt.xlabel("Version")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "02_version_countplot.png"), dpi=300)
plt.close()


plt.figure(figsize=(8, 5))
sns.histplot(data=plot_df, x="longitude", bins=40, kde=True)
plt.title("Map 3: Longitude Distribution")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "03_longitude_distribution.png"), dpi=300)
plt.close()


plt.figure(figsize=(8, 5))
sns.histplot(data=plot_df, x="latitude", bins=40, kde=True)
plt.title("Map 4: Latitude Distribution")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "04_latitude_distribution.png"), dpi=300)
plt.close()


plt.figure(figsize=(8, 5))
sns.histplot(data=plot_df, x="height_km", hue="version_plot", bins=40, kde=True)
plt.title("Map 5: Height Distribution by Version")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "05_height_distribution_by_version.png"), dpi=300)
plt.close()


plt.figure(figsize=(8, 5))
sns.histplot(data=plot_df, x="velocity_kms", hue="version_plot", bins=40, kde=True)
plt.title("Map 6: Velocity Distribution by Version")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "06_velocity_distribution_by_version.png"), dpi=300)
plt.close()


plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=plot_df,
    x="longitude",
    y="latitude",
    hue="version_plot",
    alpha=0.75
)
plt.title("Map 7: Longitude vs Latitude")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "07_longitude_vs_latitude_scatter.png"), dpi=300)
plt.close()


plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=plot_df,
    x="height_km",
    y="velocity_kms",
    hue="version_plot",
    alpha=0.75
)
plt.title("Map 8: Height vs Velocity")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "08_height_vs_velocity_scatter.png"), dpi=300)
plt.close()


plt.figure(figsize=(9, 5))
sns.boxplot(
    data=plot_df,
    x="version_plot",
    y="height_km",
    hue="version_plot",
    legend=False
)
plt.title("Map 9: Height Boxplot by Version")
plt.xlabel("Version")
plt.ylabel("Height km")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "09_height_boxplot_by_version.png"), dpi=300)
plt.close()


plt.figure(figsize=(7, 5))
corr_data = plot_df[available_numeric].corr()
sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Map 10: Numeric Feature Correlation")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "10_correlation_heatmap.png"), dpi=300)
plt.close()


print("10 seaborn maps saved in:", plot_dir)


print("\nStarting Machine Learning using 8 models...")

target_col = "version"
feature_cols = ["longitude", "latitude", "height_km", "velocity_kms"]

missing_required = [col for col in [target_col] + feature_cols if col not in df.columns]

if missing_required:
    raise ValueError("Missing required columns: " + str(missing_required))


ml_df = df.dropna(subset=[target_col]).copy()
ml_df = ml_df.dropna(subset=feature_cols, how="all")

class_counts = ml_df[target_col].value_counts()
valid_classes = class_counts[class_counts >= 2].index
ml_df = ml_df[ml_df[target_col].isin(valid_classes)].copy()

X = ml_df[feature_cols]
y = ml_df[target_col]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM": SVC(kernel="rbf", class_weight="balanced", probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42)
}


results = []
fitted_models = {}


for model_name, model in models.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Accuracy": acc
    })

    fitted_models[model_name] = pipe

    print("\n" + "=" * 70)
    print(model_name)
    print("Accuracy:", round(acc * 100, 2), "%")

    print(classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    ))


results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
results_path = os.path.join(output_dir, "ml_model_accuracy_results.csv")
results_df.to_csv(results_path, index=False)


print("\n" + "=" * 70)
print("Machine Learning Accuracy Ranking")
print(results_df)


plt.figure(figsize=(10, 5))
sns.barplot(
    data=results_df,
    x="Accuracy",
    y="Model",
    hue="Model",
    legend=False
)
plt.title("8 Machine Learning Models Accuracy")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ml_accuracy_comparison.png"), dpi=300)
plt.close()


best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
best_pred = best_model.predict(X_test)


cm = confusion_matrix(
    y_test,
    best_pred,
    labels=np.arange(len(label_encoder.classes_))
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

disp.plot(xticks_rotation=45)
plt.title("Best ML Model Confusion Matrix: " + best_model_name)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_ml_confusion_matrix.png"), dpi=300)
plt.close()


if JOBLIB_AVAILABLE:
    joblib.dump(best_model, os.path.join(model_dir, "best_ml_model.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "version_label_encoder.pkl"))
    print("\nBest ML model saved:", os.path.join(model_dir, "best_ml_model.pkl"))


print("\nBest ML Model:", best_model_name)
print("Best Accuracy:", round(float(results_df.iloc[0]["Accuracy"]) * 100, 2), "%")


print("\nStarting Deep Learning Time Series Models: RNN, LSTM, GRU...")


try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(42)
    np.random.seed(42)

    ts_feature_cols = ["longitude", "latitude", "velocity_kms"]
    ts_target_col = "height_km"

    ts_df = df[ts_feature_cols + [ts_target_col]].dropna().copy()
    ts_df = ts_df.reset_index(drop=True)

    if len(ts_df) < 100:
        print("Not enough complete rows for deep learning time series.")
    else:
        X_ts_raw = ts_df[ts_feature_cols].values
        y_ts_raw = ts_df[[ts_target_col]].values

        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()

        X_ts_scaled = x_scaler.fit_transform(X_ts_raw)
        y_ts_scaled = y_scaler.fit_transform(y_ts_raw)

        def make_sequences(X_data, y_data, sequence_length):
            X_seq, y_seq = [], []

            for i in range(sequence_length, len(X_data)):
                X_seq.append(X_data[i - sequence_length:i])
                y_seq.append(y_data[i])

            return np.array(X_seq), np.array(y_seq)

        sequence_length = 10

        X_seq, y_seq = make_sequences(
            X_ts_scaled,
            y_ts_scaled,
            sequence_length
        )

        split_index = int(len(X_seq) * 0.80)

        X_train_seq = X_seq[:split_index]
        X_test_seq = X_seq[split_index:]

        y_train_seq = y_seq[:split_index]
        y_test_seq = y_seq[split_index:]

        def build_deep_model(layer_type):
            model = Sequential()
            model.add(Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
            model.add(layer_type(64, activation="tanh"))
            model.add(Dropout(0.20))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(1))

            model.compile(
                optimizer="adam",
                loss="mse",
                metrics=["mae"]
            )

            return model

        dl_models = {
            "RNN": build_deep_model(SimpleRNN),
            "LSTM": build_deep_model(LSTM),
            "GRU": build_deep_model(GRU)
        }

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        dl_results = []

        for name, model in dl_models.items():
            print("\n" + "=" * 70)
            print("Training", name)

            history = model.fit(
                X_train_seq,
                y_train_seq,
                epochs=30,
                batch_size=32,
                validation_split=0.20,
                callbacks=[early_stop],
                verbose=1
            )

            pred_scaled = model.predict(X_test_seq)

            pred = y_scaler.inverse_transform(pred_scaled)
            actual = y_scaler.inverse_transform(y_test_seq)

            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)

            dl_results.append({
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "R2_Score": r2
            })

            print(name, "RMSE:", round(rmse, 4))
            print(name, "MAE:", round(mae, 4))
            print(name, "R2 Score:", round(r2, 4))

            model.save(os.path.join(
                model_dir,
                name.lower() + "_time_series_model.keras"
            ))

            comparison_count = min(200, len(actual))

            plt.figure(figsize=(12, 5))
            plt.plot(actual[:comparison_count], label="Actual Height")
            plt.plot(pred[:comparison_count], label="Predicted Height")
            plt.title(name + " Time Series Prediction: Actual vs Predicted Height")
            plt.xlabel("Time Step")
            plt.ylabel("Height km")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(
                output_dir,
                name.lower() + "_actual_vs_predicted.png"
            ), dpi=300)
            plt.close()

        dl_results_df = pd.DataFrame(dl_results).sort_values(by="RMSE")
        dl_results_path = os.path.join(
            output_dir,
            "deep_learning_time_series_results.csv"
        )

        dl_results_df.to_csv(dl_results_path, index=False)

        print("\n" + "=" * 70)
        print("Deep Learning Time Series Results")
        print(dl_results_df)

        print("\nDeep learning models saved in:", model_dir)

except ImportError:
    print("\nTensorFlow is not installed, so RNN/LSTM/GRU training was skipped.")
    print("Install it with: pip install tensorflow")


print("\nFinal assessment part 2 question 2 completed('I work 3 days in this part').")

# This code cleans and studies SpaceX Starlink satellite data using Pandas, NumPy, and Seaborn.
# It creates 10 maps/plots such as missing values heatmap, version count, longitude/latitude distributions, height/velocity plots, scatter plots, boxplot, and correlation heatmap.
# It trains 8 machine learning models: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, and AdaBoost.
# It also uses deep learning models RNN, LSTM, and GRU to predict satellite height using time series data.