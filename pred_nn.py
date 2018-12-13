import prediction as p
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn1 = KNeighborsClassifier(p.neighbors)
knn1.fit(p.x_train, p.y_train)
kprediction = accuracy_score(p.y_test, knn1.predict(p.x_test))
print("NN Accuracy: " + str(kprediction))