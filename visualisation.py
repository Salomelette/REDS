import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

train = list(csv.reader(open("higgsb/training.csv","r"), delimiter=','))

xs = np.array([list(map(float, row[1:-2])) for row in train[1:]])
ys = np.array([row[-1] for row in train[1:]])


(numPoints,numFeatures) = xs.shape

xs = np.add(xs, np.random.normal(0.0, 0.0001, xs.shape))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(xs)

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_scaled)
print(X_pca.shape)

ex_variance = np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio) 

Xax = np.arange(0,250000)
Yax = X_pca[:,0]

cdict = {'s':'red','b':'green'}
laby = {'s':'s','b':'b'}
marker = {'s':'*','b':'o'}
alpha = {'s':.3, 'b':.5}

fig,ax = plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')

for y in np.unique(ys):
 ix = np.where(ys==y)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[y],s=40,
           label=labl[y],marker=marker[y],alpha=alpha[y])


plt.ylabel("Principal Component",fontsize=14)
plt.legend()
plt.show()


