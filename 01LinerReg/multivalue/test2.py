# Predict the salary of the employ based on the data
import pandas as pd

data = pd.read_csv("hiring.csv")

# Data processing
data.experience = data.experience.fillna("zero")

import math

median_test_score = math.floor(data["test_score(out of 10)"].mean())

data["test_score(out of 10)"] = data["test_score(out of 10)"].fillna(median_test_score)

# Converting the words in in the dataframe to the number
# Requirement: pip install word2number
from word2number import w2n

data.experience = data.experience.apply(w2n.word_to_num)

X = data[["experience", "test_score(out of 10)", "interview_score(out of 10)"]]
y = data["salary($)"]


# Making the model
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, y)
model.coef_
model.intercept_

# Doing Predications
model.predict([[2, 9, 6], [12, 10, 10]])  # [[experience, test_score, interview_score]]
