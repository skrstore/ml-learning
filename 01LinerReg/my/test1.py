import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6])
y = np.array([273, 274, 275, 276, 277, 278, 279])


import LinearReg as model

b = model.fit(x, y)

x_new = np.array([11, 13, 15])
y_pred = model.predict(x_new)

print(y_pred)

# plotting regression line
model.plot_linear_reg(x, y, x_new, y_pred)
