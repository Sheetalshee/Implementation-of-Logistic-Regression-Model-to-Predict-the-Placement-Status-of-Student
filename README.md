# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. finally execute the program and display the output.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHEETAL.R
RegisterNumber: 212223230206

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset=pd.read_csv("/content/Placement_Data_Full_Class (1).csv")
dataset.head()
```
![image](https://github.com/user-attachments/assets/cf8a7981-4a78-4139-a4f7-7e805a165801)

```
dataset.tail()
```
![image](https://github.com/user-attachments/assets/90fa2593-5070-44c2-aff8-01e6e8e2f215)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/1756b603-2ad3-4f09-9937-692da74b0e1a)
```
dataset=dataset.drop('sl_no',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
```
![image](https://github.com/user-attachments/assets/c25292fe-796a-4671-9841-48fc146e0bb2)
```
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
```
![image](https://github.com/user-attachments/assets/33f432c8-5372-4098-b7b9-539ca5aaacd6)

```
dataset.info()
```
![image](https://github.com/user-attachments/assets/1ef044f5-3c79-4386-a19f-c5f289973e8e)
```
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
clf=LogisticRegression()
clf.fit(x_train,y_train)
```

![image](https://github.com/user-attachments/assets/8476198b-dfb7-4a5a-aff0-0d1fb6067020)

```
y_pred=clf.predict(x_test)
clf.score(x_test,y_test)
```
![image](https://github.com/user-attachments/assets/580075e1-1e6c-4d5c-b394-8f3f10b43de8)
```
from sklearn.metrics import  accuracy_score, confusion_matrix
cf=confusion_matrix(y_test, y_pred)
cf
```

![image](https://github.com/user-attachments/assets/5732a1c2-fc6b-4be9-8d2f-7a64d323f5e6)
```
accuracy=accuracy_score(y_test, y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/09d1a12a-cee3-4f16-bf92-1de7f83bb68f)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
