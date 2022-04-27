weather = [
    "Sunny",
    "Sunny",
    "Overcast",
    "Rainy",
    "Rainy",
    "Rainy",
    "Overcast",
    "Sunny",
    "Sunny",
    "Rainy",
    "Sunny",
    "Overcast",
    "Overcast",
    "Rainy",
]
temp = [
    "Hot",
    "Hot",
    "Hot",
    "Mild",
    "Cool",
    "Cool",
    "Cool",
    "Mild",
    "Cool",
    "Mild",
    "Mild",
    "Mild",
    "Hot",
    "Mild",
]

play = [
    "No",
    "No",
    "Yes",
    "Yes",
    "Yes",
    "No",
    "Yes",
    "No",
    "Yes",
    "Yes",
    "Yes",
    "Yes",
    "Yes",
    "No",
]


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# Converting string labels into numbers.
weather_encoded = le.fit_transform(weather)
le.classes_  # ['Overcast', 'Rainy', 'Sunny']
temp_encoded = le.fit_transform(temp)
le.classes_ # ['Cool', 'Hot', 'Mild']
label = le.fit_transform(play)
le.classes_ # ['No', 'Yes']

features = list(zip(weather_encoded, temp_encoded))
features

from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(features, label)

# Predict Output
predicted = model.predict([[0, 2]])  # 0:Overcast, 2:Mild
model.score(features, label)
predicted
