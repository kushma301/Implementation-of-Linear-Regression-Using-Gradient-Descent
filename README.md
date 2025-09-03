# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: S.KUSHMA
RegisterNumber: 212224040168  
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("C:\\Users\\admin\\Downloads\\50_Startups.csv")
data.head()
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
## Output:

## Data information
<img width="830" height="186" alt="Screenshot 2025-09-03 155339" src="https://github.com/user-attachments/assets/c48bae19-693d-4700-9c18-40b00683aabe" />

## Value of X
<img width="738" height="947" alt="image" src="https://github.com/user-attachments/assets/bc6814b1-e107-477c-a0a1-aee3ffa146e9" />

## Value of X1
<img width="556" height="1017" alt="image" src="https://github.com/user-attachments/assets/a6332cd1-339d-47a6-bc79-fd92dc080ba4" />

## Predicted value
<img width="1446" height="52" alt="image" src="https://github.com/user-attachments/assets/76b355d2-586a-4660-8825-0c8e936463eb" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
