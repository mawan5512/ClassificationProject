# 3d Projection

import prediction as p
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
onehotX = p.encoded_np
principalComponents = pca.fit_transform(onehotX)
principalDf = p.pd.DataFrame(data=principalComponents
                             , columns=['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = p.pd.concat([principalDf, p.Y], axis=1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_zlabel('Principal Component 3', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = ['NO', '>30', '<30']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['readmitted'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
plt.show()
