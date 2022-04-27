import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
dir(iris)

df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
df["target"] = iris["target"]


df["flower_name"] = df.target.apply(lambda x: iris.target_names[x])
print(df.head())

print(df[45:55])

df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

import matplotlib.pyplot as plt

# %matplotlib inline

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(
    df0["sepal length (cm)"], df0["sepal width (cm)"], color="green", marker="+"
)
plt.scatter(df1["sepal length (cm)"], df1["sepal width (cm)"], color="blue", marker=".")

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.scatter(
    df0["petal length (cm)"], df0["petal width (cm)"], color="green", marker="+"
)
plt.scatter(df1["petal length (cm)"], df1["petal width (cm)"], color="blue", marker=".")
plt.show()


X = df.drop(["target", "flower_name"], axis="columns")
y = df.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.svm import SVC

model = SVC(gamma="auto").fit(X_train, y_train)
model.predict(X_test)

model.score(X_train, y_train)
model.score(X_test, y_test)
