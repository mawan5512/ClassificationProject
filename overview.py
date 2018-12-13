import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
from matplotlib import style
from matplotlib.pyplot import pie, axis, show

# Puts dataset into dataframe, Change filepath depending on where it is in your computer
diabetes = pd.read_csv("data/diabetic_data.csv")
# Split data into target class Y, and data attributes X
X = diabetes.loc[:, 'race':'diabetesMed']
Y = diabetes['readmitted']

# Basic count information for data set
unique1=diabetes.encounter_id.nunique()
unique2=diabetes.patient_nbr.nunique()
racecount=diabetes.groupby('race')['patient_nbr'].nunique()
print("Number of unique patients: " + str(unique2))
print("Number of unique encounters: " + str(unique1))

plt.pie(racecount, labels=racecount.index)
plt.title("Race Breakdown")
plt.show()

