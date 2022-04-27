import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Loading the data
data = load_iris()
dir(data)

X = pd.DataFrame(data["data"])
y = np.array(data["target"])

# Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


# Tranning and predictions
from sklearn.neighbors import KNeighborsClassifier

# K = 5
model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
y_pred = model.predict(X_test)
model.score(X_train, y_train)
model.score(X_test, y_test)

# K = 3
model1 = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
model1.score(X_test, y_test)
model1.score(X_train, y_train)

# evaluate the algorithm
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

