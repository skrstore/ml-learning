import numpy as np

x = np.array([5, 15, 25, 35, 45, 55, 60, 40, 33, 66, 76, 45]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38, 50, 40, 66, 76, 66, 77])

from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(x, y)

# y_new = model.predict(x_new)
# x_new = np.array([5, 15, 20, 25, 30, 50]).reshape((-1, 1))


# Saving Model using pickle
import pickle

with open("model.model", "wb") as f:
    pickle.dump(model, f)

# Using Saved model for prediction
with open("model1.model", "rb") as f:
    mp = pickle.load(f)
print(mp.predict(np.array([50]).reshape((-1, 1))))


# Saving Model using joblib
# from sklearn.externals import joblib # because this is Depreated
import joblib

joblib.dump(model, "model_joblib")

# Using Saved model for prediction
mj = joblib.load("model_joblib")
mj.predict(np.array([50]).reshape((-1, 1)))
