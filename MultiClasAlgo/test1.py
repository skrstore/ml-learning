from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
digits = load_digits()



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3)


lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)


# n_estimators is the number of trees to be used in the forest. Since in Random Forest multiple decision trees are created, this parameter is used to control the number of trees to be used in the process
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)



# KFold
# StratifiedKFlod - this will split the dataset into the train and the test fold such that the each fold will not have the same type of the category otherewise the result will not be acurate