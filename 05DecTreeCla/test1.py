# LOADING THE DATA
import pandas as pd

col_names = [
    "pregnant",
    "glucose",
    "bp",
    "skin",
    "insulin",
    "bmi",
    "pedigree",
    "age",
    "label",
]
data = pd.read_csv("diabetes.csv", skiprows=1, names=col_names)
data.head()


# split dataset in features and target variable
feature_cols = ["pregnant", "insulin", "bmi", "age", "glucose", "bp", "pedigree"]
X = data[feature_cols]  # Features
y = data.label  # Target variable
y.head()


# Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# BUILDING THE MODEL
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()  # Create Decision Tree classifer object
model.fit(X_train, y_train)  # Train Decision Tree Classifer

y_pred = model.predict(X_test)  # Predict the response for test dataset


from sklearn import metrics  # metrics module for accuracy calculation

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

