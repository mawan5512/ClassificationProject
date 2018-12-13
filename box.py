import prediction as p
import matplotlib.pyplot as plt

# This code plots the correlation between the numerical data and target class data
# Splits the numerical data bsed on which class it is in
l30 = p.num.loc[p.diabetes['readmitted'] == '<30']
g30 = p.num.loc[p.diabetes['readmitted'] == '>30']
no = p.num.loc[p.diabetes['readmitted'] == 'NO']
# plots it using a ggplot
plt.style.use('ggplot')
fig, axs = plt.subplots(3, 4, figsize=(20, 10))
for i in range(1, 12):
    plt.subplot(3, 4, i)
    plt.boxplot([l30[p.num.columns.values[i - 1]], g30[p.num.columns.values[i - 1]], no[p.num.columns.values[i - 1]]])
    plt.xticks([1, 2, 3], ['<30', '>30', 'NO'], rotation=30, ha='right')
    plt.ylabel(p.num.columns.values[i - 1])
plt.subplots_adjust(wspace=0.30, hspace=0.25)
fig.delaxes(axs[2][3])
plt.show()