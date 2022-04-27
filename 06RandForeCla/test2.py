from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# With the seed reset (every time), the same set of numbers will appear every time.
# np.random.seed(0)

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data.head()

data["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

X = data.drop("species", axis="columns")
y = iris.target


# Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5).fit(X_train, y_train)
y_test_new = model.predict(X_test)

# model evaluation
model.score(X_train, y_train)
model.score(X_test, y_test)


from sklearn.metrics import confusion_matrix, classification_report

# y_pred = model.predict(X_test)
y_pred = model.predict(X_train)
confusion_matrix(y_train, y_pred)
print(classification_report(y_train, y_pred))
