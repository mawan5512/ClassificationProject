import prediction as p
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(p.x_train, p.y_train)
dprediction = accuracy_score(p.y_test, dtc.predict(p.x_test))
print("DTC Accuracy: " + str(dprediction))