import numpy as np

c = np.arange(100).reshape(-1,1)
# f = np.array([36, 46.4, 59, 71.6])
f = 1.8 * c + 32 # traditional development

# print(c)
# print(f)

import matplotlib.pyplot as plt

# plt.scatter(c,f)
# plt.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(c,f)


print(model.coef_[0][0])
print(model.intercept_[0])



a = np.array([34, 95]).reshape(-1,1)
b = model.predict(a)

print(model.score(c,f))

print(b)