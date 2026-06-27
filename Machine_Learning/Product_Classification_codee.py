import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
df = pd.read_csv(
    r"C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\Product_Classification_code.csv",
    delimiter=","
)

# 2. Clean column names (IMPORTANT STEP)
df.columns = df.columns.str.strip()

# 3. Check required columns exist (safe check)
required_cols = ['Product Title', 'Category Label']

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column in dataset: {col}")

# 4. Drop missing values safely
df = df.dropna(subset=required_cols)

# 5. Split features and labels
X = df['Product Title']
y = df['Category Label']

# 6. TF-IDF Vectorization
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_encoded = tfidf.fit_transform(X)

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Train Model (Logistic Regression)
model = LogisticRegression(
    max_iter=1000,
    solver='saga'
)

model.fit(X_train, y_train)

# 9. Predictions
y_pred = model.predict(X_test)

# 10. Evaluation
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== Model Accuracy: {accuracy * 100:.2f}% ===\n")

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

plt.figure(figsize=(12, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=np.unique(y),
    yticklabels=np.unique(y)
)

plt.title('Product Classification Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()