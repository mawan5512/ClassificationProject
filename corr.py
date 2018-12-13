import prediction as p
import seaborn as sns
import matplotlib.pyplot as plt

##This plots the correlation between all the categorical data, look at the bottom row to see all the
##correlation with the target class the positive num means positive correlation, 0 mean no correaltion
## and negative means no correlation
# Calculate the corelation matrix
df = p.cat
corr = df.apply(lambda x: p.pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
# Plot correlation matrix on a heat map
mask = p.np.zeros_like(corr, dtype=p.np.bool)
mask[p.np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()