import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()

dir(digits)


# %matplotlib inline
import matplotlib.pyplot as plt

plt.gray()
plt.matshow(digits.images[0])
plt.show()

df = pd.DataFrame(digits.data)
df.head()

df["target"] = digits.target

X = df.drop("target", axis="columns")
y = df.target

# Splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=20).fit(X_train, y_train)

# Model evaluation
model.score(X_train, y_train)
model.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)

plt.matshow(digits.images[181])
plt.show()
y_pred = model.predict(X_test)
