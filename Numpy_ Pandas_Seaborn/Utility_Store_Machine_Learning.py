import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df =pd.read_csv(r"C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\utility_store_dataset.csv")

print("df.head():\n",df.head())
print("df.shape:\n" ,df.shape)
print(df.describe())

variables = ['Discount Percentage']

for var in variables:
    plt.figure()
    sns.regplot(x=var, y='Original Price', data=df).set(title=f'Regression plot of {var} and Original Price');
    plt.show()

read = input("Wait here: \n")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df =pd.read_csv(r"C:\Users\User\OneDrive\Documents\GitHub\desktop-tutorial\My_Ai-course-Bin\Machine_Learning\utility_store_dataset.csv")
# Display first rows
print(df.head())

item_encoder = LabelEncoder()
df['Item'] = item_encoder.fit_transform(df['Item'])

category_encoder = LabelEncoder()
df['Category'] = category_encoder.fit_transform(df['Category'])

X = df[['Item', 'Original Price', 'Discount Percentage']]

Y = df['Category']

print("\nX values:")
print(X.head())

print("\nY values:")
print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# -----------------------------
# RESULTS
# -----------------------------

print("\nAccuracy Score:")
print(accuracy_score(Y_test, Y_pred))

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))


plt.figure(figsize=(8,5))

sns.countplot(x='Category', data=df)

plt.title('Category Count Plot')

plt.show()


variables = ['Original Price', 'Discount Percentage']

for var in variables:

    plt.figure(figsize=(8,5))

    sns.regplot(
        x=var,
        y='Item',
        data=df
    ).set(title=f'Regression plot of {var} and Item')

    plt.show()