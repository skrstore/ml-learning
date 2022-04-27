import numpy as np

X = np.array(
    [
        [1, 1],
        [1.2, 2],
        [1.5, 2.5],
        [2, 2],
        [1.8, 1.2],
        [2, 3],
        [3, 3],
        [3, 2],
        [2.5, 1.5],
        [3, 1],
    ]
)
y = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0])

import matplotlib.pyplot as plt

# plt.scatter(X[:, 0], X[:, 1])
# plt.show()


from sklearn.svm import SVC

model = SVC(kernel="linear").fit(X, y)

Y_pred = model.predict(X)
Y_pred = model.predict([[1, 1]])
Y_pred = model.predict([[-1, -1], [-2, -1], [1, 1], [2, 1]])


# Seeing the Probability
model = SVC(kernel="poly", probability=True).fit(X, y)
model = SVC(kernel="sigmoid", probability=True).fit(X, y)
model = SVC(kernel="linear", probability=True).fit(X, y)
model = SVC(probability=True).fit(X, y)
model.predict_proba(X)
model.score(X, y)

model.predict(np.array([[2.3, 3.5]]))

# Kernal Mode
# linear, poly, rbf(radial basis function), and sigmoid
