import pandas as pd

data = pd.read_csv('fruit1.csv')

from sklearn.preprocessing import LabelEncoder

color_en = LabelEncoder()
label_en = LabelEncoder()

data['color_en'] = color_en.fit_transform(data['color'])
data['label_en'] = label_en.fit_transform(data['label'])

X = data.drop(['color', 'dim', 'label'], axis='columns')
y = data.label_en

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier().fit(X,y)

y_pred = model.predict([[1,2], [0, 2]])
print(y_pred)