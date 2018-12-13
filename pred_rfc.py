import prediction as p
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=28,criterion = "entropy", min_samples_split=10)
rfc.fit(p.x_train, p.y_train)
rprediction = accuracy_score(p.y_test, rfc.predict(p.x_test))
print("RFC Accuracy: " + str(rprediction))