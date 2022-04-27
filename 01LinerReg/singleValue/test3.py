import pandas as pd

data = pd.read_csv("homeprices.csv")

X = data[["area"]]
y = data.price


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
newArea = pd.read_csv("areas.csv")

from matplotlib import pyplot as plt

newPrice = model.predict(newArea)
plt.plot(newArea, newPrice)
plt.show()
