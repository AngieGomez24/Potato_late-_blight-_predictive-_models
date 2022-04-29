# -*- coding: utf-8 -*-
"""
@author: ANGIE GÓMEZ
"""
############################################
import matplotlib.pyplot as plt
plt.style.use('classic')
############################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import normalize

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
############################################
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
############################################
image_directory = '../dataset_augmented/'
SIZE = 150
dataset = []
label = []
#####################
healthy_images = os.listdir(image_directory + 'healthy/')


infected_images = os.listdir(image_directory + 'late_blight/')
for i, image_name in enumerate(infected_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'late_blight/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)
        
        
for i, image_name in enumerate(healthy_images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory + 'healthy/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)    
        
#############################################
dataset = np.array(dataset)
label = np.array(label)
print(dataset)
#############################################

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=0)
###############################################
#Normalizar valores
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)
###########################################################

INPUT_SHAPE = (SIZE, SIZE, 3)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

#####################################################

history = model.fit(X_train,
                        y_train,
                        batch_size=64,
                        verbose=1,
                        epochs = 300,
                        validation_data=(X_test, y_test),
                        shuffle=False
                    )

model.save('cnn_aug_v1.h5')

#####################################################

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


########################################################

_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ",(acc* 100.0), "%")

##########################################################
mythreshold= 0.954096
y_pred = (model.predict(X_test)>= mythreshold).astype(int)
cm=confusion_matrix(y_test, y_pred)  
print(cm)

#####################################################
#ROC
y_preds = model.predict(X_test).ravel()

fpr, tpr, thresholds = roc_curve(y_test, y_preds)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.show()

##################################################
i = np.arange(len(tpr)) 
roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'thresholds' : pd.Series(thresholds, index=i)})
ideal_roc_thresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]]  #Locate the point where the value is close to 0
print("Ideal threshold is: ", ideal_roc_thresh['thresholds']) 

####################################################
#AUC
auc_value = auc(fpr, tpr)
print("Area under curve, AUC = ", auc_value)
