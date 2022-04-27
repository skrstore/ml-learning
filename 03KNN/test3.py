# Load dataset
from sklearn import datasets

wine = datasets.load_wine()
print(wine.feature_names)
print(wine.target_names)
print(wine.data[0:5])
print(wine.target)
print(wine.data.shape)
print(wine.target.shape)
type(wine.target)
type(wine)
dir(wine)

X = wine["data"]
y = wine["target"]

# SPLITTING THE DATA
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier

# MODEL FOR K = 5
model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
y_pred = model.predict(X_test)

model.score(X_train, y_train)
model.score(X_test, y_test)

# Import scikit-learn metrics module for accuracy calculation
# from sklearn import metrics
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# MPDEL FOR K = 3
model1 = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = model1.predict(X_test)

model1.score(X_train, y_train)
model1.score(X_test, y_test)
