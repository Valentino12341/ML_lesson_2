# multi variable regression  = With more input instead of just 1 for linear equations

# linear equation  =    y = m*x+c
"""
# first   import dataset
 data preprocessing -remove.replace null values , convert all string in to integer 
data  analaysis  ip/op = input-output

split the data- train for example 80% and test with 20 
select the approptiate ML algorithm 
train the ML with the data

predict and calcukate the error and accuracy 

"""

import pandas as pd 
import numpy as np 


data = pd.read_csv("titanic.csv")

print(data.head())
print(data.info())

data["Sex"]= data["Sex"].map({"male":0,"female":1})


#x=input and y output

x=data[["Pclass","Sex","Age"]]
y= data["Survived"]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=5)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

# training the model
model.fit(x_train,y_train)
ypredict = model.predict(x_test)

from sklearn.metrics import mean_squared_error

mse1 = np.sqrt(mean_squared_error(y_test,ypredict))
print("the rmse in case of multivariable regression is ",mse1)

# with 20% its 0.376354161096018
# with 30% its 0.3759209660236956
# with 50% its 0.37692846426082016

#polynominal regression 
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2)

x_train_poly = poly_features.fit_transform(x_train)

x_test_poly = poly_features.fit_transform(x_test)


model2 = LinearRegression()
#training the model 
model2.fit(x_train_poly,y_train)
ypredict2 = model2.predict(x_test_poly)


#rmse  root meet squared error
mse2 = np.sqrt(mean_squared_error(y_test,ypredict2))
print("The rmse in case of polyvariable regression is ",mse2)




