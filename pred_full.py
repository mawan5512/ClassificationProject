import prediction as p
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

knn1 = KNeighborsClassifier(p.neighbors)
knn1.fit(p.x_train, p.y_train)
kprediction = accuracy_score(p.y_test, knn1.predict(p.x_test))
print("NN Accuracy: " + str(kprediction))

dtc = DecisionTreeClassifier()
dtc.fit(p.x_train, p.y_train)
dprediction = accuracy_score(p.y_test, dtc.predict(p.x_test))
print("DTC Accuracy: " + str(dprediction))

rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(p.x_train, p.y_train)
rprediction = accuracy_score(p.y_test, rfc.predict(p.x_test))
print("RFC Accuracy: " + str(rprediction))

mlp = MLPClassifier(max_iter=p.mlp_iter)
mlp.fit(p.x_train, p.y_train)
mprediction = accuracy_score(p.y_test, mlp.predict(p.x_test))
print("MLP Accuracy: " + str(mprediction))


