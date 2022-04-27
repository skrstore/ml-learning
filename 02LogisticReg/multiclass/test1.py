from sklearn.datasets import load_digits

digits = load_digits()
print(digits.data.shape)
dir(digits)  # ['DESCR', 'data', 'images', 'target', 'target_names']
digits.data[0]


# 8 X 8 images of the digits
digits.images[0]
digits.target_names
digits.target
digits.DESCR[0]


import matplotlib.pyplot as plt

plt.gray()
plt.matshow(digits.images[0])
plt.show()


digits.target[0:5]
digits.data[0]
len(digits.data[0])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2
)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)

model.score(X_test, y_test)

model.predict(digits.data[0:5])


# Prediction
plt.matshow(digits.images[67])
digits.target[67]


model.predict([digits.data[67]])
model.predict(digits.data[0:5])