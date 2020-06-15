# -*- coding: utf-8 -*-
"""
Created on Sat May 23 04:07:15 2020

@author: kingslayer
"""

#import libriaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("Car_Purchasing_Data.csv", encoding='latin-1')


#Visualsation
import seaborn as sns

sns.pairplot(dataset.drop(columns=["Customer Name","Customer e-mail"]))


#Splitting
dataset=dataset.drop(columns=["Customer Name","Customer e-mail","Country"])

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc_X=MinMaxScaler()
X=sc_X.fit_transform(X)
y=y.reshape(-1,1)
sc_y=MinMaxScaler()
y=sc_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


#ANN
from keras.models import Sequential
from keras.layers import Dense,Dropout

regressor=Sequential()

regressor.add(Dense(output_dim=25,init="uniform",activation="relu",input_dim=5))

regressor.add(Dense(output_dim=25,init="uniform",activation="relu"))
regressor.add(Dropout(0.2))

regressor.add(Dense(output_dim=25,init="uniform",activation="relu"))
regressor.add(Dropout(0.2))

regressor.add(Dense(output_dim=1,init="uniform",activation="linear"))

regressor.compile(optimizer="adam",loss='mean_squared_error')

reg_hist=regressor.fit(X_train,y_train,batch_size=50,epochs=200,validation_split=0.2)

y_pred=regressor.predict(X_test)



#Results
results=pd.DataFrame(y_test)
results["Prediction"]=sc_y.inverse_transform(y_pred)
results["Real"]=sc_y.inverse_transform(y_test)
results["Difference"]=results["Real"]-results["Prediction"]

#Plot
reg_hist.history.keys()

plt.plot(reg_hist.history["loss"])
plt.plot(reg_hist.history["val_loss"])
plt.title("Training and Validation Loss")
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.legend(["Training Loss","Validation Loss"])

plt.figure(figsize=(20,20))
plt.plot(results["Real"])
plt.plot(results["Prediction"])
plt.title("Real vs Prediction")
plt.xlabel("Customer")
plt.ylabel("Amount")
plt.legend(["Real","Prediction"])
