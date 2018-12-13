import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Puts dataset into dataframe, Change filepath depending on where it is in your computer
diabetes = pd.read_csv("data/diabetic_data.csv")
# Split data into target class Y, and data attributes X
X = diabetes['diabetesMed']
Y = diabetes['readmitted']
le = LabelEncoder()
X = le.fit_transform(X)
Y = le.fit_transform(Y)
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
# Split the data into training/testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()