import pandas as pd
data = pd.read_csv("salaries.csv")
data.head()

inputs = data.drop("salary_more_then_100k", axis="columns")
target = data["salary_more_then_100k"]


# converting the labeled data into the values
from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
inputs["company_n"] = le_company.fit_transform(inputs["company"])
inputs["job_n"] = le_company.fit_transform(inputs["job"])
inputs["degree_n"] = le_company.fit_transform(inputs["degree"])
inputs.head()


input_n = inputs.drop(["company", "job", "degree"], axis="columns")
input_n.head()


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier().fit(input_n, target)
model.score(input_n, target)

model.predict([[2, 2, 1]])
model.predict([[2, 0, 1]])

