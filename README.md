# Implementation-of-Logistic-Regression-using-sklearn
# Datasetslink:https://drive.google.com/drive/folders/15XG8HzPdMaWgGYv5DGG4uN4KL00Nebt1?usp=share_link
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score 
dataset = pd.read_csv('Social_Network_Ads.csv') 
print(dataset.columns)
X = dataset[['Age', 'EstimatedSalary']] 
y = dataset['Purchased']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25) 
#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test) 
classifier = LogisticRegression() 
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test) 
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred)) 
print("Accuracy Score",accuracy_score(y_test, y_pred)) 
