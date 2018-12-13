import prediction as p
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(max_iter=p.mlp_iter)
mlp.fit(p.x_train, p.y_train)
mprediction = accuracy_score(p.y_test, mlp.predict(p.x_test))
print("MLP Accuracy: " + str(mprediction))