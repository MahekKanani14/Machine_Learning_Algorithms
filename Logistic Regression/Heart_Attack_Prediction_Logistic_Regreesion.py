import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Loading Dataset into pandas dataframe
df = pd.read_csv("framingham.csv")

# Getting Info Of Dataset
print(df.info())

# Dropping Missing or inappropriate values
df = df.dropna()

# Separating target and features
y = df['TenYearCHD']
x = df.drop(columns='TenYearCHD')

# splitting dataset into Training and Testing set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=14)

# Z-score normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Loss Function
def loss_func(y_pred,y_true):
    e = 1e-9 # adding e prevent log argument from being zero
    return -np.mean(((1-y_true)*np.log(1-y_pred+e) + (y_true)*(np.log(y_pred + e))))

# Sigmoid Function
def sigmoid(y):
    return 1/(1+np.exp(-y))


# Gredient Decent for Finding Optimal Weights,Basis
def gredient_decent(weight,basis,x,y,learning_rate,iter):
   
    for i in range(iter):
        lin_pred = np.dot(x,weight) + basis
        y_pred = sigmoid(lin_pred)
        
        gd_w = (1/(len(x)))*np.dot(x.T,(y_pred - y))
        gd_b = (1/(len(x)))*np.sum(y_pred - y)

        weight -= learning_rate*gd_w
        basis  -= learning_rate*gd_b

    return weight,basis


# Training Model
w = np.zeros(x_train.shape[1])
b = 0
w,b = gredient_decent(w,b,x_train,y_train,0.005,10000)

# Testing Of Model
test_predictions = sigmoid(np.dot(x_test, w) + b)

# setting 0.5 as threshold
log_accuracy = accuracy_score(y_test,(test_predictions>=0.5))

#Accuracy Of Model
print(f"Using logistic regression we get an accuracy of {round(log_accuracy*100,2)}%")