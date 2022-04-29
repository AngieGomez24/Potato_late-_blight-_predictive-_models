# -*- coding: utf-8 -*-
"""
@author: ANGIE GÓMEZ
"""

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

os.chdir("D:\Proyecto")
os.getcwd()

filename = "dataset.csv"
raw_data = open(filename)
df = pd.read_csv(raw_data,delimiter=";", skiprows=1, index_col='index', 
                 names=['index','MeanB','StdB','VarB','RankB','MeanG','StdG','VarG','RankG','MeanR','StdR','VarR',
                                'RankR','MeanH','StdH','VarH','RankH','MeanS','StdS','VarS','RankS','MeanV','StdV','VarV',
                                'RankV','Contrast_0','Dissimilarity_0','Homogeneity_0','Energy_0','Correlation_0','ASM_0',
                                'Contrast_pi_4','Dissimilarity_pi_4','Homogeneity_pi_4','Energy_pi_4','Correlation_pi_4','ASM_pi_4',
                                'Contrast_pi_2','Dissimilarity_pi_2','Homogeneity_pi_2','Energy_pi_2','Correlation_pi_2','ASM_pi_2',
                                'Contrast_3pi_4','Dissimilarity_3pi_4','Homogeneity_3pi_4','Energy_3pi_4','Correlation_3pi_4','ASM_3pi_4','Diagnosis'])

features = ['MeanB','StdB','VarB','RankB','MeanG','StdG','VarG','RankG','MeanR','StdR','VarR',
               'RankR','MeanH','StdH','VarH','RankH','MeanS','StdS','VarS','RankS','MeanV','StdV','VarV',
               'RankV','Contrast_0','Dissimilarity_0','Homogeneity_0','Energy_0','Correlation_0','ASM_0',
               'Contrast_pi_4','Dissimilarity_pi_4','Homogeneity_pi_4','Energy_pi_4','Correlation_pi_4','ASM_pi_4',
               'Contrast_pi_2','Dissimilarity_pi_2','Homogeneity_pi_2','Energy_pi_2','Correlation_pi_2','ASM_pi_2',
               'Contrast_3pi_4','Dissimilarity_3pi_4','Homogeneity_3pi_4','Energy_3pi_4','Correlation_3pi_4','ASM_3pi_4']

# separación de caracteristicas
x = df.loc[:, features].values
print(x)

# separación de etiquetas
y = df.loc[:,['Diagnosis']].values
print(y)

# Estandarización de caracteristicas
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['Diagnosis']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Sana', 'Infectada']
colors = ['r', 'g',]
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Diagnosis'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#Salida: carateristicas PCA 2 componentes
with open('pca_data.csv', 'w') as file:
    principalDf.to_csv(file,sep=';',decimal='.',index=True)