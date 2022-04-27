# Given the home prices find out price of a home that has
# 3000 sqr ft area, 3 bedrooms, 40 years old
# 2500 sqr ft area, 4 bedrooms, 5 years old

import pandas as pd
data = pd.read_csv("homeprices.csv")

# For data processing(or data cleaning) (because the data is missing in actual data)
import math
median = math.floor(data.bedrooms.median())
data.bedrooms = data.bedrooms.fillna(median)

X = data[["area", "bedrooms", "age"]]
y = data.price

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
# .fit(independent variables , target variable)
print(model.coef_)  # array([  112.06244194, 23388.88007794, -3231.71790863])
print(model.intercept_)  # 221323.00186540443

newPrice = model.predict([[3000, 3, 40]])


# 112.06244194*3000 + 23388.88007794*3 + -3231.71790863*40 + 221323.00186540384
