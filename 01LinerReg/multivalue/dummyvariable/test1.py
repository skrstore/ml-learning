import pandas as pd

data = pd.read_csv("homeprices.csv")

dummies = pd.get_dummies(data.town)

merged = pd.concat([data, dummies], axis="columns")
final = merged.drop(["town", "west windsor"], axis="columns")

X = final.drop("price", axis="columns")
y = final.price

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)

out = model.predict([[2800, 0, 1]])
out = model.predict([[3400, 0, 0]])
out = model.predict([[2600, 1, 0]])

print(model.score(X, y))