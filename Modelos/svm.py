# -*- coding: utf-8 -*-
"""
@author: ANGIE GÓMEZ
"""
import pandas as pd
import os
import joblib

########################################
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt

os.chdir("D:\Proyecto")
os.getcwd()

filename = "processed_dataset.csv"
raw_data = open(filename)
df = pd.read_csv(raw_data,delimiter=";", skiprows=1, index_col='index', 
                 names=['index','MeanB','StdB','VarB','RankB','MeanG','StdG','VarG','RankG','MeanR','StdR','VarR',
                                'RankR','MeanH','StdH','VarH','RankH','MeanS','StdS','VarS','RankS','MeanV','StdV','VarV',
                                'RankV','Contrast_0','Dissimilarity_0','Homogeneity_0','Energy_0','Correlation_0','ASM_0',
                                'Contrast_pi_4','Dissimilarity_pi_4','Homogeneity_pi_4','Energy_pi_4','Correlation_pi_4','ASM_pi_4',
                                'Contrast_pi_2','Dissimilarity_pi_2','Homogeneity_pi_2','Energy_pi_2','Correlation_pi_2','ASM_pi_2',
                                'Contrast_3pi_4','Dissimilarity_3pi_4','Homogeneity_3pi_4','Energy_3pi_4','Correlation_3pi_4','ASM_3pi_4',
                                'Component_1','Component_2','Diagnosis'])

#Conversión de etiquetas no numericas a numericas.
df.Diagnosis[df.Diagnosis == 'Infectada'] = 0
df.Diagnosis[df.Diagnosis == 'Sana'] = 1


#Definición de variable dependiente
Y = df["Diagnosis"]
Y=Y.astype('int')

#Definición de vaiables independientes
"""
#caracteristicas de color
X = df.drop(labels = ['Contrast_0','Dissimilarity_0','Homogeneity_0','Energy_0','Correlation_0','ASM_0',
               'Contrast_pi_4','Dissimilarity_pi_4','Homogeneity_pi_4','Energy_pi_4','Correlation_pi_4','ASM_pi_4',
               'Contrast_pi_2','Dissimilarity_pi_2','Homogeneity_pi_2','Energy_pi_2','Correlation_pi_2','ASM_pi_2',
               'Contrast_3pi_4','Dissimilarity_3pi_4','Homogeneity_3pi_4','Energy_3pi_4','Correlation_3pi_4','ASM_3pi_4',
               'Component_1','Component_2','Diagnosis'], axis=1) 
#print(X.head())

"""
"""
#caracteristicas de textura
X = df.drop(labels = ['MeanB','StdB','VarB','RankB','MeanG','StdG','VarG','RankG','MeanR','StdR','VarR',
               'RankR','MeanH','StdH','VarH','RankH','MeanS','StdS','VarS','RankS','MeanV','StdV','VarV',
               'RankV','Component_1','Component_2','Diagnosis'], axis=1) 
#print(X.head())

"""
"""
#caracteristicas PCA
X = df.drop(labels = ['MeanB','StdB','VarB','RankB','MeanG','StdG','VarG','RankG','MeanR','StdR','VarR',
               'RankR','MeanH','StdH','VarH','RankH','MeanS','StdS','VarS','RankS','MeanV','StdV','VarV',
               'RankV','Contrast_0','Dissimilarity_0','Homogeneity_0','Energy_0','Correlation_0','ASM_0',
               'Contrast_pi_4','Dissimilarity_pi_4','Homogeneity_pi_4','Energy_pi_4','Correlation_pi_4','ASM_pi_4',
               'Contrast_pi_2','Dissimilarity_pi_2','Homogeneity_pi_2','Energy_pi_2','Correlation_pi_2','ASM_pi_2',
               'Contrast_3pi_4','Dissimilarity_3pi_4','Homogeneity_3pi_4','Energy_3pi_4','Correlation_3pi_4','ASM_3pi_4',
               'Diagnosis'], axis=1) 
#print(X.head())

"""
#caracteristicas combinadas
X = df.drop(labels = ['Diagnosis'], axis=1) 

#División de datos en entrenamiento y validación
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

#Instanciar modelo
SVM_model = svm.SVC(kernel='linear')
SVM_model.fit(X_train, y_train)

prediction_test = SVM_model.predict(X_test)

# Guardar
joblib.dump(SVM_model, "./SVM_combined_v1.joblib", compress=3)

#Exactitud
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))


# Generar matriz de confusión
matrix = plot_confusion_matrix(SVM_model, X_test, y_test,
                                 cmap=plt.cm.Blues)
plt.title('Confusion matrix for our classifier SVM combined')
plt.show(matrix)
plt.show()


#Curva ROC
y_preds = SVM_model.predict(X_test).ravel()

fpr, tpr, thresholds = roc_curve(y_test, prediction_test)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

#AUC
auc_value = auc(fpr, tpr)
print("Area under curve, AUC = ", auc_value)


#loaded_rf = joblib.load("./SVM_color_v1.joblib")
#prediction = loaded_rf.predict(X_test)
#print(y_test, prediction)

