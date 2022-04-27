import numpy as np
from matplotlib import pyplot as plt

x = np.array([5, 15, 25, 35, 45, 55, 60, 40, 33, 66, 76, 45]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38, 50, 40, 66, 76, 66, 77])

plt.scatter(x, y, label="Data 1", color="k", s=25, marker="o")
plt.plot(x, y, color="green")


from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(x, y)
model.score(x, y)  # score
model.intercept_  # intercept
model.coef_  # slope


y_pred = model.predict(x)
plt.scatter(x, y_pred, label="Data 1", color="k", s=25, marker="o")
y_pred = model.intercept_ + model.coef_ * x
plt.plot(y_pred, x, color="green")

x_new = np.array([5, 15, 20, 25, 30, 50]).reshape((-1, 1))
y_new = model.predict(x_new)
plt.scatter(x_new, y_new, label="Google", color="k", s=25, marker="o")
plt.plot(y_new, x_new, color="blue")

print(y_new)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
plt.show()