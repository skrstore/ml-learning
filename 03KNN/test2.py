# Assigning features and label variables
# First Feature
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
# Second Feature
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

# Label or target varible
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
temp_encoded = le.fit_transform(temp)
# combinig weather and temp into single list of tuples

features = list(zip(weather_encoded, temp_encoded))
label = le.fit_transform(play)


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3).fit(features, label)
predicted = model.predict([[0, 2]])  # 0:Overcast, 2:Mild
print(predicted)

model.score(features, label)