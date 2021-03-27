#Md Fahim Faez Abir
#2018-1-60-032
import pandas as pd
import numpy as np
import math
import random
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv(r"C:\Users\HP\Desktop\regression\Fish.csv")

datas = data.iloc[:,:].values

labelencoder_datas = LabelEncoder()
row = data.shape[0]
col = data.shape[1]


n = 0
mark = 0
column=[]
for i in data:
    if data[i].dtype=='object':
        column.append(n)
        mark=1
    n = n+1
if mark==1:
    length = len(column)
    for t in range(0,length):
        for j in range(0,col):
            if j == column[t]:
                datas[:,j]= labelencoder_datas.fit_transform(datas[:,j])
                    
y_exp = np.ones(row)
for i in range(0,row):
    y_exp[i] = datas[i,col-1]
    #print(y_exp)               
x =np.ones((row,col))
for i in range(0,row):
    for j in range(1,col):
        x[i,j] = datas[i,j-1]
    #print(x) 

w = np.ones(col)
    #print(w)
for i in range(0,col):
    w[i] = 0.5
    #print(w)
iteration = 1
for i in range(0,15):
    l_r = random.uniform(0,1)
    Y = np.zeros(row)
        #dot_multiplication
    for i in range(0,row):
        for j in range(0,col):
            Y[i] += x[i,j]*w[j]
            
    E = np.ones(row)
        #print(Y)
    for j in range(0,row):
        E[j] = Y[j]-y_exp[j]
        #print(E,"\n")
    w_new = np.ones(col)
    X_T = np.ones((col,row))
        #Transpose
    for i in range(0,row):
        for j in range(0,col):
            X_T[j,i] = x[i,j]
        
    p = np.zeros(col)
        #dot_multiplication
    for i in range(0,col):
        for j in range(0,row):
            p[i] += X_T[i,j]*E[j]
        
    for i in range(0,col):
        p[i]*=l_r
    for i in range(0,col):
        w_new[i] = w[i]-p[i]
        #w_new = np.subtract(w,np.multiply(l_r,np.dot(X_T,E)))
        # print("w_new:\n",w_new)
    for j in range(0,col):
        w[j] = w_new[j]
    
    e = 0.0
    for j in range(0,row):
        e+= pow((Y[j]-y_exp[j]),2)
    
    print("i = ",iteration," l_r= ",l_r," e = ",e/2)    
    iteration=iteration+1    
