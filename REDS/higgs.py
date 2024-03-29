import random,string,math,csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb 
import seaborn as sns
from sklearn.model_selection import train_test_split


train = list(csv.reader(open("higgsb/training.csv","r"), delimiter=','))

print("Nombre d'exemples dans le train :",len(train))
xs = np.array([list(map(float, row[1:-2])) for row in train[1:]])
ys = np.array([row[-1] for row in train[1:]])


(numPoints,numFeatures) = xs.shape

xs = np.add(xs, np.random.normal(0.0, 0.0001, xs.shape))

sSelector = np.array([row[-1] == 's' for row in train[1:]])
bSelector = np.array([row[-1] == 'b' for row in train[1:]])
#for s in sSelector: 
#    if s: 
#        print(s)
#        pdb.set_trace()
print("Nombre de labels s : " ,len(np.where(sSelector==True)[0]))
print("Nombre de label b : ",len(np.where(bSelector==True)[0]))
weights = np.array([float(row[-2]) for row in train[1:]])
sumWeights = np.sum(weights)
sumSWeights = np.sum(weights[sSelector])
sumBWeights = np.sum(weights[bSelector])
print("Pourcentage de la classe s: : ", len(np.where(sSelector==True)[0])/len(train[1:])*100)
print("Pourcentage de la classe b: : ", len(np.where(bSelector==True)[0])/len(train[1:])*100)

#print("Pourcentage du poids s: : ", sumSWeights/sumWeights)
#print("Pourcentage du poids b: : ", sumBWeights/sumWeights)

train = pd.read_csv("higgsb/training.csv") 

print(train.describe())

print(train.describe(include=['object']))
print(train.describe(include='all'))

#
#plt.matshow(train.corr())
#plt.show()

sns.heatmap(train.corr(), annot=False)
fig, ax = plt.subplots()
train['Label'].value_counts().plot(ax=ax, kind='bar')

x_train, x_test, y_train, y_test = train_test_split(xs,ys)

        
        
def plus_freq(x_train, y_train): 
    sSelector = np.where(y_train == 's')
    bSelector = np.where(y_train == 'b')
    p_s = len(sSelector)/len(y_train)
    p_b = len(bSelector)/len(y_train)
    if p_s > p_b: 
        return 's'
    else: 
        return 'b'

def random_clf(x_train, y_train):
    res = []    
    

    
