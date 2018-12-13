import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing  import StandardScaler
from sklearn.feature_selection import VarianceThreshold


# Control Variables
testperc = 0.1
neighbors = 3
mlp_iter = 1000
var_tresh = 1

lb = LabelBinarizer()
le = LabelEncoder()
one = OneHotEncoder(sparse=True)
sel = VarianceThreshold(threshold=(var_tresh * (1 - var_tresh)))

# Import Data File
diabetes = pd.read_csv("/Users/mohammedawan/Downloads/cs5810-project-master/data/diabetic_data.csv")
df = diabetes

# dropping discharge_disposition_id = 11, which means the patient died
# dropping the missing values in gender
# drop_Idx = set(df['race'][df['race'] == '?'].index)
drop_Idx = set(df[(df['diag_1'] == '?') & (df['diag_2'] == '?') & (df['diag_3'] == '?')].index)
drop_Idx = drop_Idx.union(set(df[df['discharge_disposition_id'] == 11].index))
drop_Idx = drop_Idx.union(set(df['gender'][df['gender'] == 'Unknown/Invalid'].index))
new_Idx = list(set(df.index) - set(drop_Idx))
df = df.iloc[new_Idx]
# dropping columns with too many missing values
df = df.drop(['weight', 'payer_code', 'medical_specialty'], axis = 1)
# remove columns having same value in each row: citoglipton, examide
df = df.drop(['citoglipton', 'examide'], axis = 1)

#Combines the medication to one column
keys = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
        'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide-metformin', 'tolazamide',
        'metformin-pioglitazone','metformin-rosiglitazone', 'glimepiride-pioglitazone',
        'glipizide-metformin', 'troglitazone', 'tolbutamide', 'acetohexamide']
for col in keys:
    colname = str(col) + 'temp'
    df[colname] = df[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)

df['numchange'] = 0

for col in keys:
    colname = str(col) + 'temp'
    df['numchange'] = df['numchange'] + df[colname]
    del df[colname]

Y = df['readmitted']
num = df.select_dtypes(include='int64')
cat = df.select_dtypes(include='object')

# Split data into target class Y, and data attributes X
#X = diabetes.loc[:, 'race':'diabetesMed']
#Y = diabetes['readmitted']
# Split the into nominal and numerical attributes
#num = X.select_dtypes(include='int64')
#cat = X.select_dtypes(include='object')

lb.fit(Y)
binarized_Y = lb.transform(Y)

encoded = []
cat_arr = cat.values
for i in range(cat_arr.shape[1]):
    le.fit(cat_arr[:, i])
    encoded.append(le.transform(cat_arr[:, i]))
encoded_np = np.array(encoded).T
onehotX = one.fit_transform(encoded_np)

# append numerical data so that you can have one dataset
num_arr = StandardScaler().fit_transform(num.values)
for i in range(num_arr.shape[1]):
    np.append(onehotX, num_arr[:, i])
onehotX = sel.fit_transform(onehotX)

# Split Data
x_train, x_test, y_train, y_test = train_test_split(onehotX, binarized_Y, test_size=testperc)

x_train = x_train.todense()
x_test = x_test.todense()

print("Current Shape: " + str(onehotX.shape))
#print("Current Variables:")
#print("Test Data Percentage: " + str(testperc))
#print("Number of Neighbors: " + str(neighbors))
#print("MLP Iterations: "+ str(mlp_iter))
#print("Variance Threshold: " + str(var_tresh))