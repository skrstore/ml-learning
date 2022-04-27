# Multi lables

from sklearn import datasets
wine = datasets.load_wine()

wine.feature_names
wine.target_names
wine.data.shape
wine.data[0:5]
wine.target


# Split dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109)


from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

model.score(X_train, y_train)
model.score(X_test, y_test)



from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))