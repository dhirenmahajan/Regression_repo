# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 23:39:42 2022

@author: dhire
"""
import pandas as pd 
from sklearn.datasets import fetch_california_housing 

cali = fetch_california_housing() 
cali_df = pd.DataFrame(cali.data, columns=cali.feature_names) 
cali_df['MedHouseValue'] = pd.Series(cali.target) 

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( cali.data, cali.target, random_state=11) 

from sklearn.linear_model import LinearRegression 

mu_regress = LinearRegression() 
mu_regress.fit(X=X_train, y=y_train) 
predicted = mu_regress.predict(X_test) 
expected = y_test 

from sklearn import metrics


print("Multiple Linear Regression using all features")
X = cali_df.iloc[:,0:8]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression() 
mu_regress.fit(x_train,y_train)
y_predict = mu_regress.predict(x_test)
print("R2 score: ",metrics.r2_score(y_test,y_predict))
print("MSE score: ",metrics.mean_squared_error(y_test,y_predict))
print("\n")

X = cali_df[['MedInc']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 0 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("          has MSE score: ",metrics.mean_squared_error(y_test,y_pred))



X = cali_df[['HouseAge']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 1 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("          has MSE score:  ",metrics.mean_squared_error(y_test,y_pred))


X = cali_df[['AveRooms']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 2 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("          has MSE score: ",metrics.mean_squared_error(y_test,y_pred))


X = cali_df[['AveBedrms']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 3 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("          has MSE score: ",metrics.mean_squared_error(y_test,y_pred))


X = cali_df[['Population']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 4 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("          has MSE score: ",metrics.mean_squared_error(y_test,y_pred))


X = cali_df[['AveOccup']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 5 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("          has MSE score: ",metrics.mean_squared_error(y_test,y_pred))

X = cali_df[['Latitude']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 6 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("          has MSE score: ",metrics.mean_squared_error(y_test,y_pred))


X = cali_df[['Longitude']]
Y = cali_df[['MedHouseValue']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
mu_regress = LinearRegression()
mu_regress.fit(x_train,y_train)
y_pred = mu_regress.predict(x_test)
print("Feature 7 has R2 score: ",metrics.r2_score(y_test,y_pred))
print("          has MSE score: ",metrics.mean_squared_error(y_test,y_pred))


"""Multiple linear regression is a more specific calculation than simple 
linear regression. For straight-forward relationships, 
simple linear regression may easily capture the relationship between the two variables. 
For more complex relationships requiring more consideration, 
multiple linear regression is often better.
"""