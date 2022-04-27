# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
plt.scatter(X,y)
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)


X[0]
X_poly[0]


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_poly, y)


out = model.predict(X_poly)
model.intercept_, model.coef_

plt.scatter(X, out)
plt.scatter(X,y)
plt.show()
