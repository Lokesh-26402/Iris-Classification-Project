import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['species'] = iris.target

print(df.head())

print(df.info())

print(df.describe())

print(df['species'].value_counts())

sns.pairplot(df, hue='species')

plt.show()

plt.figure(figsize=(8,6))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.show()

X = df.drop('species', axis=1)

y = df['species']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))

print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

cm = confusion_matrix(y_test, knn_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("KNN Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
print(classification_report(y_test, knn_pred))
joblib.dump(knn, 'iris_model.pkl')

model = joblib.load('iris_model.pkl')

sample = [[5.1, 3.5, 1.4, 0.2]]

sample = scaler.transform(sample)

prediction = model.predict(sample)

print("Predicted Class:", prediction[0])

print("Species:", iris.target_names[prediction][0])

model = joblib.load('iris_model.pkl')

sample = [[5.1, 3.5, 1.4, 0.2]]

sample = scaler.transform(sample)

prediction = model.predict(sample)

print("Predicted Class:", prediction[0])

print("Species:", iris.target_names[prediction][0])