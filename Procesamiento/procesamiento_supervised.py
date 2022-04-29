# -*- coding: utf-8 -*-
"""
@author: ANGIE GÓMEZ
"""


import numpy as np
import pandas as pd
import cv2
import os
import json
import random
import uuid

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt

# Preprocesado y modelado
# ==============================================================================
from skimage.feature import greycomatrix, greycoprops


from datetime import date



def read_images_healthy():
    path_dir = os.getcwd()
    path_images = os.path.join(path_dir, 'dataset/healthy')
    full_path_images = []
    images_name = os.listdir(path_images)
    for image in images_name:
        full_path_images.append(os.path.join(path_images, image))
    return full_path_images

def read_images_late_blight():
    path_dir = os.getcwd()
    path_images = os.path.join(path_dir, 'dataset/late_blight')
    full_path_images = []
    images_name = os.listdir(path_images)
    for image in images_name:
        full_path_images.append(os.path.join(path_images, image))
    return full_path_images

def metadata(image):
    path_dir = os.getcwd()
    path_images = os.path.join(path_dir, image)

    image_data = os.path.split(path_images)
    image_name = os.path.splitext(image_data[1])

    date1 = date.today()

    fuente = 'autores'

    metadata = {
      "image": image_name[0],
      "format": image_name[1],
      "date": date1.strftime('%Y/%m/%d'),
      "source": fuente
    }
        
    return metadata

def crop(source):
    tipos_cultivo = ['invernadero','campo_abierto']
    especies = ['solanum_tuberosum','solanum_phureja']
    variedades = ['pastusa','r12','tocarrena']

    if source == 'autores':  
        tipo_cultivo = "campo_abierto"
        especie = "solanum_tuberosum"
        variedad = "pastusa"
        edad = "5.5"
        localidad = "Aquitania"
        temperatura = "10"
        humedad = "94%"
        precipitacion = "285"
        altura = "3080"
    else:
        tipo_cultivo = tipos_cultivo[random.randint(0,len(tipos_cultivo)-1)]
        especie = especies[random.randint(0,len(especies)-1)]
        variedad =  'amarilla' if especie == 'solanum_phureja' else variedades[random.randint(0,len(variedades)-1)]
        edad = random.randint(1,8)
        localidad = 'NA'
        temperatura = random.randint(8,16)
        humedad = f"{random.randint(60,80)}%"   
        precipitacion = random.randint(46,56)
        altura = random.randint(2600,3600)
        
    crop = {
        "type": tipo_cultivo,
        "specie": especie,
        "variety": variedad,
        "age": edad,
        "location": localidad,
        "temperature": temperatura,
        "humidity": humedad,
        "precipitation": precipitacion,
        "height": altura
    }
    
    return crop

def segmentation(img):
    #hsv
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    th_low_green = np.array([30, 30, 0])
    th_high_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(img_hsv, th_low_green, th_high_green)
    img_mask = cv2.bitwise_and(img, img, mask = green_mask)
    th_low_brown = np.array([9, 50, 50])
    th_high_brown = np.array([25, 255, 200])
    brown_mask = cv2.inRange(img_hsv, th_low_brown, th_high_brown)
    new_img = cv2.bitwise_and(img, img, mask = brown_mask)
    opor = cv2.bitwise_or(img_mask,new_img)
    plt.subplot(1, 4, 1)
    plt.imshow(opor)
    plt.subplot(1, 4, 2)
    plt.imshow(img_mask)
    plt.subplot(1, 4, 3)
    plt.imshow(new_img)
    plt.subplot(1, 4, 4)
    plt.imshow(img)
    plt.show()
    
    return [opor,green_mask]
    

def colorFeatures(img,n):
    df = pd.DataFrame()
    df = df.assign(index=None)
    df = df.assign(MeanB=None)
    df = df.assign(StdB=None)
    df = df.assign(VarB=None)
    df = df.assign(RankB=None)
    df = df.assign(MeanG=None)
    df = df.assign(StdG=None)
    df = df.assign(VarG=None)
    df = df.assign(RankG=None)
    df = df.assign(MeanR=None)
    df = df.assign(StdR=None)
    df = df.assign(VarR=None)
    df = df.assign(RankR=None)
    df = df.assign(MeanH=None)
    df = df.assign(StdH=None)
    df = df.assign(VarH=None)
    df = df.assign(RankH=None)
    df = df.assign(MeanS=None)
    df = df.assign(StdS=None)
    df = df.assign(VarS=None)
    df = df.assign(RankS=None)
    df = df.assign(MeanV=None)
    df = df.assign(StdV=None)
    df = df.assign(VarV=None)
    df = df.assign(RankV=None)
    b, g, r = cv2.split(img) 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    nueva_fila = pd.Series([n,np.mean(b), np.std(b), np.var(b),np.max(b)-np.min(b),np.mean(g), np.std(g), np.var(g),np.max(g)-np.min(g), np.mean(r), np.std(r), np.var(r),np.max(r)-np.min(r),np.mean(h), np.std(h), np.var(h),np.max(h)-np.min(h),np.mean(s), np.std(s), np.var(s),np.max(s)-np.min(s), np.mean(v), np.std(v), np.var(v),np.max(v)-np.min(v)], index=df.columns) 
    df = df.append(nueva_fila, ignore_index=True)

    features = {
          "mean": {
              "r": np.mean(r),
              "b": np.mean(b),
              "g": np.mean(g),
              "h": np.mean(h),
              "s": np.mean(s),
              "v": np.mean(v),
          },
          "std": {
              "r": np.std(r),
              "b": np.std(b),
              "g": np.std(g),
              "h": np.std(h),
              "s": np.std(s),
              "v": np.std(v),
          },
          "var": {
              "r": np.var(r),
              "b": np.var(b),
              "g": np.var(g),
              "h": np.var(h),
              "s": np.var(s),
              "v": np.var(v),
          },
          "rank": {
              "r": int(np.max(r)-np.min(r)),
              "b": int(np.max(b)-np.min(b)),
              "g": int(np.max(g)-np.min(g)),
              "h": int(np.max(h)-np.min(h)),
              "s": int(np.max(s)-np.min(s)),
              "v": int(np.max(v)-np.min(v)),
          }
    }
  
    return [features, df]

#segmentacion por texturas
def contrast_feature(matrix_coocurrence):
    contrast = greycoprops(matrix_coocurrence, 'contrast')
    return contrast

def dissimilarity_feature(matrix_coocurrence):
    dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')
    return dissimilarity

def homogeneity_feature(matrix_coocurrence):
    homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
    return homogeneity

def energy_feature(matrix_coocurrence):
    energy = greycoprops(matrix_coocurrence, 'energy')
    return energy

def correlation_feature(matrix_coocurrence):
    correlation = greycoprops(matrix_coocurrence, 'correlation')
    return correlation

def asm_feature(matrix_coocurrence):
    asm = greycoprops(matrix_coocurrence, 'ASM')
    return asm

def textureFeatures(grayImg,n):
    df = pd.DataFrame()
    df = df.assign(index="None")
    df = df.assign(Contrast_0="None")
    df = df.assign(Dissimilarity_0="None")
    df = df.assign(Homogeneity_0="None")
    df = df.assign(Energy_0="None")
    df = df.assign(Correlation_0="None")
    df = df.assign(ASM_0="None")
    df = df.assign(Contrast_pi_4="None")
    df = df.assign(Dissimilarity_pi_4="None")
    df = df.assign(Homogeneity_pi_4="None")
    df = df.assign(Energy_pi_4="None")
    df = df.assign(Correlation_pi_4="None")
    df = df.assign(ASM_pi_4="None")
    df = df.assign(Contrast_pi_2="None")
    df = df.assign(Dissimilarity_pi_2="None")
    df = df.assign(Homogeneity_pi_2="None")
    df = df.assign(Energy_pi_2="None")
    df = df.assign(Correlation_pi_2="None")
    df = df.assign(ASM_pi_2="None")
    df = df.assign(Contrast_3pi_4="None")
    df = df.assign(Dissimilarity_3pi_4="None")
    df = df.assign(Homogeneity_3pi_4="None")
    df = df.assign(Energy_3pi_4="None")
    df = df.assign(Correlation_3pi_4="None")
    df = df.assign(ASM_3pi_4="None")

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
    inds = np.digitize(grayImg, bins)
    max_value = inds.max()+1
    matrix_coocurrence = greycomatrix(inds, #Numpy matrix for co-occurrence matrix calculation
                                      [1],#Step
                                      [0, np.pi/4, np.pi/2, 3*np.pi/4],#Direction angle
                                      levels=max_value, #Co-occurrence matrix order
                                      normed=False, symmetric=False)
    contrast = contrast_feature(matrix_coocurrence)
    dissimilarity = dissimilarity_feature(matrix_coocurrence)
    homogeneity = homogeneity_feature(matrix_coocurrence)
    energy = energy_feature(matrix_coocurrence)
    correlation = correlation_feature(matrix_coocurrence)
    asm = asm_feature(matrix_coocurrence)
    nueva_fila = pd.Series([n,contrast[0][0], contrast[0][1], contrast[0][2],contrast[0][3],
                                dissimilarity[0][0],  dissimilarity[0][1],  dissimilarity[0][2], dissimilarity[0][3], 
                                homogeneity[0][0], homogeneity[0][1], homogeneity[0][2],homogeneity[0][3],
                                energy[0][0],  energy[0][1],  energy[0][2], energy[0][3],
                                correlation[0][0], correlation[0][1], correlation[0][2],correlation[0][3], 
                                asm[0][0],  asm[0][1],  asm[0][2], asm[0][3]], index=df.columns) 
    df = df.append(nueva_fila, ignore_index=True)
   
    features = {
          "contrast": {
              "0": contrast[0][0],
              "pi_4": contrast[0][1],
              "pi_2": contrast[0][2],
              "3pi_4": contrast[0][3]  
          },
          "dissimilarity":{
              "0": dissimilarity[0][0],
              "pi_4": dissimilarity[0][1],
              "pi_2": dissimilarity[0][2],
              "3pi_4": dissimilarity[0][3]  
          },
          "homogeneity":{
              "0": homogeneity[0][0],
              "pi_4": homogeneity[0][1],
              "pi_2": homogeneity[0][2],
              "3pi_4": homogeneity[0][3] 
          },
          "energy":{
              "0": energy[0][0],
              "pi_4": energy[0][1],
              "pi_2": energy[0][2],
              "3pi_4": energy[0][3] 
          },
          "correlation":{
              "0": correlation[0][0],
              "pi_4": correlation[0][1],
              "pi_2": correlation[0][2],
              "3pi_4": correlation[0][3] 
          },
          "asm":{
              "0": asm[0][0],
              "pi_4": asm[0][1],
              "pi_2": asm[0][2],
              "3pi_4": asm[0][3] 
          }
    }
    return [features,df]

def dataObject(uuidImage,metadata, crop, diagnosis):
    data = {
        "uuid": str(uuidImage),
        "metadata": metadata,
        "crop": crop,
        "diagnosis": diagnosis
    }
    
    return data

def dataImages():
    images_healthy = read_images_healthy()
    images_late_blight = read_images_late_blight()  
    dataImages = []
    for image in images_healthy:
        uuidImage = uuid.uuid4()
        md = metadata(image)
        cp = crop(md['source'])
        dataImages.append(dataObject(uuidImage,md,cp, 'sana')) 
    
    for image in images_late_blight:
        uuidImage = uuid.uuid4()
        md = metadata(image)
        cp = crop(md['source'])
        dataImages.append(dataObject(uuidImage,md,cp, 'infectada')) 
    return dataImages

def processedDataImages():
    images_healthy = read_images_healthy()
    images_late_blight = read_images_late_blight()

    dataset = pd.DataFrame(columns=['index','MeanB','StdB','VarB','RankB','MeanG','StdG','VarG','RankG','MeanR','StdR','VarR',
                                    'RankR','MeanH','StdH','VarH','RankH','MeanS','StdS','VarS','RankS','MeanV','StdV','VarV',
                                    'RankV','Contrast_0','Dissimilarity_0','Homogeneity_0','Energy_0','Correlation_0','ASM_0',
                                    'Contrast_pi_4','Dissimilarity_pi_4','Homogeneity_pi_4','Energy_pi_4','Correlation_pi_4','ASM_pi_4',
                                    'Contrast_pi_2','Dissimilarity_pi_2','Homogeneity_pi_2','Energy_pi_2','Correlation_pi_2','ASM_pi_2',
                                    'Contrast_3pi_4','Dissimilarity_3pi_4','Homogeneity_3pi_4','Energy_3pi_4','Correlation_3pi_4','ASM_3pi_4','Diagnosis'])
    n = -1
    for image in images_healthy:
        n = n + 1
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        SIZE = 256
        dim = (SIZE, SIZE)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        seg = segmentation(resized)
        
        #cv2.imwrite('./plant_village_without_background/Potato___healthy/'+metadata(image)['image']+'.JPG',seg[0])
        
        cf = colorFeatures(seg[0],n)
        grayImg = cv2.cvtColor(seg[0], cv2.COLOR_BGR2GRAY)
        tf = textureFeatures(grayImg,n)
        
        df = pd.DataFrame()
        df = df.assign(index="None")
        df = df.assign(Diagnosis="None")
        nueva_fila = pd.Series([n, 'Sana'], index=df.columns)
        df = df.append(nueva_fila, ignore_index=True)
        
        dftomerge = pd.merge(cf[1],tf[1],on='index',how='inner')
        
        dataset = pd.concat([dataset,pd.merge(dftomerge,df,on='index',how='inner')])
    
    
    for image in images_late_blight:
        n = n + 1
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        width = 256
        height = 256
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        seg = segmentation(resized)
        
        #cv2.imwrite('./plant_village_without_background/Potato___Late_blight/'+metadata(image)['image']+'.JPG',seg[0])

        cf = colorFeatures(seg[0],n)
        grayImg = cv2.cvtColor(seg[0], cv2.COLOR_BGR2GRAY)
        tf = textureFeatures(grayImg,n)
        
        df = pd.DataFrame()
        df = df.assign(index="None")
        df = df.assign(Diagnosis="None")
        nueva_fila = pd.Series([n, 'Infectada'], index=df.columns)
        df = df.append(nueva_fila, ignore_index=True)
        
        dftomerge = pd.merge(cf[1],tf[1],on='index',how='inner')
        
        dataset = pd.concat([dataset,pd.merge(dftomerge,df,on='index',how='inner')])
    
    return [dataset]


# Metadata
imagesData = dataImages()


with open('data.json', 'w') as file:
    json.dump(imagesData, file, indent=2)
    
    
#ProcessedData
dfFeatures = processedDataImages()
with open('dataset.csv', 'w') as file:
    dfFeatures[0].to_csv(file,sep=';',decimal='.',index=False)
    
#processedDataImages()
    
##################################################################################################    
#PCA
def pca():
    os.chdir("D:\Proyecto")
    os.getcwd()
    
    filename = "pca_data.csv"
    raw_data = open(filename)
    df = np.loadtxt(raw_data, delimiter=";",skiprows=1)
    #print(df.shape)
    #print(df)
    pca = pd.DataFrame(df,columns=['index','Component_1','Component_2'])
    
    return pca

#Dataframe PCA
pca = pca()
#Dataframe ProcessedData
processedDataSet = pd.DataFrame(columns=['index','MeanB','StdB','VarB','RankB','MeanG','StdG','VarG','RankG','MeanR','StdR','VarR',
                                    'RankR','MeanH','StdH','VarH','RankH','MeanS','StdS','VarS','RankS','MeanV','StdV','VarV',
                                    'RankV','Contrast_0','Dissimilarity_0','Homogeneity_0','Energy_0','Correlation_0','ASM_0',
                                    'Contrast_pi_4','Dissimilarity_pi_4','Homogeneity_pi_4','Energy_pi_4','Correlation_pi_4','ASM_pi_4',
                                    'Contrast_pi_2','Dissimilarity_pi_2','Homogeneity_pi_2','Energy_pi_2','Correlation_pi_2','ASM_pi_2',
                                    'Contrast_3pi_4','Dissimilarity_3pi_4','Homogeneity_3pi_4','Energy_3pi_4','Correlation_3pi_4','ASM_3pi_4',
                                    'Component_1','Component_2','Diagnosis'])
processedDataSet = pd.concat([processedDataSet,pd.merge(dfFeatures[0],pca,on='index',how='inner')])

with open('processed_dataset.csv', 'w') as file:
    processedDataSet.to_csv(file,sep=';',decimal='.',index=False)


def convertDataToJson(processedDataSet):
    data = []
    for index in range(len(processedDataSet)):
        #print(index)
        #print(imagesData[index]['uuid'])
        processedData={
        "uuid": imagesData[index]['uuid'],
        "diagnosis": {
            "status": processedDataSet.iloc[index,51],
            },
        "features": {
            "component_1": processedDataSet.iloc[index,49],
            "component_2": processedDataSet.iloc[index,50]
            },
        "color_features": {
            "mean": {
              "r": processedDataSet.iloc[index,9],
              "b": processedDataSet.iloc[index,1],
              "g": processedDataSet.iloc[index,5],
              "h": processedDataSet.iloc[index,13],
              "s": processedDataSet.iloc[index,17],
              "v": processedDataSet.iloc[index,21],
          },
          "std": {
              "r": processedDataSet.iloc[index,10],
              "b": processedDataSet.iloc[index,2],
              "g": processedDataSet.iloc[index,6],
              "h": processedDataSet.iloc[index,14],
              "s": processedDataSet.iloc[index,18],
              "v": processedDataSet.iloc[index,22],
          },
          "var": {
              "r": processedDataSet.iloc[index,11],
              "b": processedDataSet.iloc[index,3],
              "g": processedDataSet.iloc[index,7],
              "h": processedDataSet.iloc[index,15],
              "s": processedDataSet.iloc[index,19],
              "v": processedDataSet.iloc[index,23],
          },
          "rank": {
              "r": processedDataSet.iloc[index,12],
              "b": processedDataSet.iloc[index,4],
              "g": processedDataSet.iloc[index,8],
              "h": processedDataSet.iloc[index,16],
              "s": processedDataSet.iloc[index,20],
              "v": processedDataSet.iloc[index,24],
          },
        },
        "texture_features": {
            "contrast": {
              "0": processedDataSet.iloc[index,25],
              "pi_4": processedDataSet.iloc[index,31],
              "pi_2": processedDataSet.iloc[index,37],
              "3pi_4": processedDataSet.iloc[index,43],
          },
          "dissimilarity":{
              "0": processedDataSet.iloc[index,26],
              "pi_4": processedDataSet.iloc[index,32],
              "pi_2": processedDataSet.iloc[index,38],
              "3pi_4": processedDataSet.iloc[index,44],
          },
          "homogeneity":{
              "0": processedDataSet.iloc[index,27],
              "pi_4": processedDataSet.iloc[index,33],
              "pi_2": processedDataSet.iloc[index,39],
              "3pi_4": processedDataSet.iloc[index,45],
          },
          "energy":{
              "0": processedDataSet.iloc[index,28],
              "pi_4": processedDataSet.iloc[index,34],
              "pi_2": processedDataSet.iloc[index,40],
              "3pi_4": processedDataSet.iloc[index,46],
          },
          "correlation":{
              "0": processedDataSet.iloc[index,29],
              "pi_4": processedDataSet.iloc[index,35],
              "pi_2": processedDataSet.iloc[index,41],
              "3pi_4": processedDataSet.iloc[index,47],
          },
          "asm":{
              "0": processedDataSet.iloc[index,30],
              "pi_4": processedDataSet.iloc[index,36],
              "pi_2": processedDataSet.iloc[index,42],
              "3pi_4": processedDataSet.iloc[index,48]
          }
         }
        }
        data.append(processedData)
    
    return data
    
processedDataJson = convertDataToJson(processedDataSet)
with open('processedData.json', 'w') as file:
    json.dump(processedDataJson, file, indent=2) 

    