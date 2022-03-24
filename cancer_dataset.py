#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:31:25 2022

@author: devrodrigosoares

BASE DE DADOS CANCER DE MAMA - > Previsão de classificação binária para o tipo de câncer de mama

INFORMACOES:
    COLUNAS: 32 -> 
                    [0] -> IDENTIFICADOR
                    [1] -> CLASSES
                    [2:] -> CARACTERISTICAS - AMOSTRAS

"""

import pandas as pd
import statistics
import seaborn as sns
from matplotlib import pyplot as plt

lKNN_AC = []
lKNN_REC = []
lKNN_PREC = []
lKNN_F1 = []

lSVM_AC = []
lSVM_REC = []
lSVM_PREC = []
lSVM_F1 = []

lMLP_AC = []
lMLP_REC = []
lMLP_PREC = []
lMLP_F1 = []


cancer_dataset = pd.read_csv('cancer_dataset.csv')

#Slicing the database between features and classes.

cancerData = cancer_dataset.iloc[:,2:31]
diagnosisClasses = cancer_dataset.iloc[:,1:2]

#Analyzing the data

#cancerData.info()
descriptiveStatistics = cancerData.describe()
valueNull = cancerData.isnull().values.any()
#print("\nA base de dados Cancer de Mama possui valores nulos? ", valueNull, "\n")

#Normalizing numerical scales in 0 and 1

from sklearn.preprocessing import MinMaxScaler

scales = MinMaxScaler()
scales.fit(cancerData)
cancerDataSetNormalized = scales.transform(cancerData)

#Training data and classes

for x in range (20):

    from sklearn.model_selection import train_test_split
    dataCancer_train, dataCancer_test, classeCancer_train, classeCancer_test = train_test_split(cancerDataSetNormalized, diagnosisClasses, test_size=0.2)
    
    #print(classeCancer_train)
    #print(classeCancer_train.shape)
    
    #KNN technique
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV 
    
    
    
    parameters = [{"n_neighbors": [1,3,5,7,9,11], "metric": ["euclidean","manhattan","mahalanobis"]}]
    
    knn = KNeighborsClassifier()
    classifier = GridSearchCV(knn, parameters,cv = 5)
    classifier.fit(dataCancer_train, classeCancer_train)
    prevision = classifier.predict(dataCancer_test)
    
    #print("\n\nTécnica KNN\n")
    
    #print(prevision)
    #print(classeCancer_test)
    
    
    #print("As melhores métricas são: ",classifier.best_params_, "\n")
    
    #print("\n\n-----------\n\n")
    
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
    acc = accuracy_score(classeCancer_test, prevision)
    f1 = f1_score(classeCancer_test, prevision, average='macro')
    prec = precision_score(classeCancer_test, prevision, average='macro')
    recall = recall_score(classeCancer_test, prevision, average='macro')
    
    #print("Acuracia: ", str(acc))
    #print("Precisao: ", str(prec))
    #print("F1: ", str(f1))
    #print("Recall: ", str(recall))
    
    lKNN_AC.append(acc)
    lKNN_PREC.append(prec)
    lKNN_REC.append(recall)
    lKNN_F1.append(f1)
    
    #print("\n--------------------------\n\n")
    
    #SVM technique
    
    from sklearn.svm import SVC
    
    parametersSVM = [{"kernel": ["poly", "rbf", "sigmoid"]}]
    svm = SVC()
    classifierSVM = GridSearchCV(svm, parametersSVM,cv = 5)
    classifierSVM.fit(dataCancer_train, classeCancer_train)
    previsionSVM = classifierSVM.predict(dataCancer_test)
    
    #print("\n\nTécnica SVM\n")
    
    #print("As melhores métricas são: ", classifierSVM.best_params_, "\n")
    
    #print(previsionSVM)
    
    accSVM = accuracy_score(classeCancer_test, previsionSVM)
    f1SVM = f1_score(classeCancer_test, previsionSVM, average='macro')
    precSVM = precision_score(classeCancer_test, previsionSVM, average='macro')
    recallSVM = recall_score(classeCancer_test, previsionSVM, average='macro')
    
    #print("\n\n-----------\n\n")
    
    
    #print("Acuracia: ", str(accSVM))
    #print("Precisao: ", str(precSVM))
    #print("F1: ", str(f1SVM))
    #print("Recall: ", str(recallSVM))
    
    lSVM_AC.append(accSVM)
    lSVM_PREC.append(precSVM)
    lSVM_REC.append(recallSVM)
    lSVM_F1.append(f1SVM)
    
    
    #print("\n--------------------------\n\n")
    
    #MLP technique
    
    from sklearn.neural_network import MLPClassifier
    
    parametersMLP = [{"activation": ["identity", "logistic", "tanh", "relu"], "hidden_layer_sizes": [(3,3)], "learning_rate_init": [0.0001, 0.001, 0.01]}]
    MLP = MLPClassifier()
    classifierMLP= GridSearchCV(MLP, parametersMLP, cv = 5)
    classifierMLP.fit(dataCancer_train, classeCancer_train)
    previsionMLP = classifierMLP.predict(dataCancer_test)
    
    #print("\n\nTécnica MLP\n")
    
    #print("As melhores métricas são: ", classifierMLP.best_params_, "\n")
    
    #print(previsionMLP)
    
    accMLP = accuracy_score(classeCancer_test, previsionMLP)
    f1MLP = f1_score(classeCancer_test, previsionMLP, average='macro')
    precMLP = precision_score(classeCancer_test, previsionMLP, average='macro')
    recallMLP = recall_score(classeCancer_test, previsionMLP, average='macro')
    
    #print("\n\n-----------\n\n")
    
    
    #print("Acuracia: ", str(accMLP))
    #print("Precisao: ", str(precMLP))
    #print("F1: ", str(f1MLP))
    #print("Recall: ", str(recallMLP))
    
    lMLP_AC.append(accMLP)
    lMLP_PREC.append(precMLP)
    lMLP_REC.append(recallMLP)
    lMLP_F1.append(f1MLP)
    
    #print("\n--------------------------\n\n")
    
#medias

mediaKNNacc = round(statistics.mean(lKNN_AC),2);
mediaKNNprec = round(statistics.mean(lKNN_PREC),2);
mediaKNNrec = round(statistics.mean(lKNN_REC),2);
mediaKNNf1 = round(statistics.mean(lKNN_F1),2);

listaKNN = []

mediaSVMacc = round(statistics.mean(lSVM_AC),2);
mediaSVMprec = round(statistics.mean(lSVM_PREC),2);
mediaSVMrec = round(statistics.mean(lSVM_REC),2);
mediaSVMf1 = round(statistics.mean(lSVM_F1),2);

mediaMLPacc = round(statistics.mean(lMLP_AC),2);
mediaMLPprec = round(statistics.mean(lMLP_PREC),2);
mediaMLPrec = round(statistics.mean(lMLP_REC),2);
mediaMLPf1 = round(statistics.mean(lMLP_F1),2);

    













