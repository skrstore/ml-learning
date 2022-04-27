# prediting we have apple or banana using the given weight and texture of the items

from sklearn.tree import DecisionTreeClassifier

# 1 ==>"smooth"
# 0 ==> "bumpy"
# 0 ==> apple
# 1 ==> orange
# Weight 
# texture

features = [[140, 1], [130, 1], [150, 0], [170, 0]]

labels = [0, 0, 1, 1]

clf = DecisionTreeClassifier()
clf = clf.fit(features, labels)

res = clf.predict([[150, 0],[130, 0],[155, 1],[165, 1]])

clf.score(features,labels)

print(res)

for i in res:
    if i== 1:
        print('Orange ', i)
    if i == 0:
        print('Apple ', i)
