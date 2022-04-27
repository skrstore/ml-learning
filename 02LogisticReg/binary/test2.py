import pandas as pd

# Data
data = pd.read_csv('insurance_data.csv')

# Making MOdel
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(data[['age']], data.bought_insurance)

# Doing Prediction
newAge = pd.read_csv('ages.csv')
newIns = model.predict(newAge)