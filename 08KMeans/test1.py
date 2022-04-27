# Q - Divide the persons into three clusters on the basis of their income
import pandas as pd
df = pd.read_csv("income.csv")
df.head()


from matplotlib import pyplot as plt
plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()


from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted


df['cluster']=y_predicted
df.head()
ot1 = df[df.cluster == 0]
plt.scatter(ot1.Age,ot1['Income($)'], marker='+')
ot2 = df[df.cluster == 1]
plt.scatter(ot2.Age,ot2['Income($)'], marker='+')


km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted
