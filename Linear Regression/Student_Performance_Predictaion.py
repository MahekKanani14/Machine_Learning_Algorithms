import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# loading dataset
df = pd.read_csv('Student_performance.csv') 

# data check
print(df.isnull().sum())
print(df.info())

# separating target and features
y = df['Performance Index']
x = df.drop(columns='Performance Index')

# encoding Yes as 1 and No as 0
x['Extracurricular Activities'] = x['Extracurricular Activities'].map({'Yes' : 1,'No' : 0})

# splitting dataset into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 2)

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Mean squared error
def costfunc(x_train,y_train,w,b):
    y_pre = np.dot(x_train,w) + b
    return np.mean((y_pre - y_train)**2)

# Gradient Descent For Linear Regression
def Gradient_Descent(x_train,y_train,weight,basis,l_rate):
    
    y_pre = np.dot(x_train,weight) + basis

    w_gredient = (2/len(x_train))*np.dot(x_train.T,y_pre - y_train)
    b_gredient = (2/len(x_train))*np.sum(y_pre - y_train)


    weight -= l_rate*w_gredient
    basis  -= l_rate*b_gredient

    return weight,basis

# Training Model Using Gradient Descent  
weight = np.zeros(x_train.shape[1])
b = 0
epochs = 10000
learning_rate = 0.001
for i in range(epochs):
    weight,b = Gradient_Descent(x_train,y_train,weight,b,learning_rate)
    if(i%1000 == 0):
        print(f"epochs : {i}  cost :  {costfunc(x_train,y_train,weight,b)}")

# Predicting value for test data
y_pre = np.dot(x_test,weight) + b

# Mean Squared Error on Test data  
print(np.mean((y_pre - y_test)**2))
