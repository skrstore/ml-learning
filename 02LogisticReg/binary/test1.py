import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv("insurance_data.csv")
plt.scatter(df.age, df.bought_insurance, color="k", marker="o")


# Splitting the data set into the training and testing data 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    df[["age"]], df.bought_insurance, test_size=0.1
)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, Y_train)


Y_test_new = model.predict(X_test)
plt.plot(X_test, Y_test_new, color='red')


# to show the acciureacy of the model
model.score(X_test, Y_test) 
model.score(X_test, Y_test_new)

model.predict_proba(X_test) # to show the probability of the test case, in the result the first column has the probability of not buying the insurance and the 2 is the probability to take th e insurance

plt.xlabel("Age")
plt.ylabel("Insurece")
plt.title("Logistic Regression")
plt.legend()
plt.show()


# NOTE: Coollection of the Data Sets
# https://www.kaggle.com/datasets