# House prediction using Linear Regression Machine learning implementation
import ssl
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#Lets load the california house pricing dataset

from sklearn.datasets import fetch_california_housing
ssl._create_default_https_context = ssl._create_unverified_context
cali = fetch_california_housing()

# # Printing dataset information
# print(f"Type of dataset{type(cali)}") 
# print(f"keys of dataset {cali.keys()}")
# print(f"description: {cali}")
# print(f"Data {cali.data}")

# Lets create dataset

dataset = pd.DataFrame(cali.data, columns=cali.feature_names)
# print(dataset.head())  # It will give first 5 records

# Lets add the dependent feature price too
dataset['price']=cali.target

# print(dataset.head())
# print(dataset.info()) # will give the datatype of dataset parameters

# Summarizing the stats of the data
print(dataset.describe())

# Check the missing values

# print(dataset.isnull().sum()) # No  missing values yet

# Exploratory data analysis
# Important step : run correlation on Linear regression problem to check how output is correlated to inputs

print(dataset.corr()) #  More negatively correlated means negative the parameter will impact decrease the house price, positive parameters increase the house price, As an example if MedInc si increasing or positively corelated means house price will increase too

# # based on this corelation we can do scater plot too
# print(sns.pairplot(dataset))

plt.scatter(dataset['MedInc'], dataset['price'])
# plt.show() It will show the plot

# We can display multiple plots to check the corelation using seaborn but we will not do that 

# Independent and dependent features
# Price is dependent feature, rest are independent feature
x=dataset.iloc[:,:-1] # will give independent feature
y=dataset.iloc[:,-1]

print(x.head())

# Train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=42)

print("x- train", x_train) # similarly we can check y_train, y_test
# To check how model is performing, I will check the x-test data
print("x=-test", x_test)

# Before model training we need to do standard scaling

# Every feature is calculated with different-different units, we need to use standard scaling to make it global minima or we can say same unit(standardize or normalize)

# Standardize the dataset because internally we use gradient descent

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test) # test data we don't need to fit transform

pickle.dump(scaler,open('scaling.pkl', 'wb'))

print(x_train)

# Implement Linear regression algorithm (Model training)
from sklearn.linear_model import LinearRegression
regression=LinearRegression()
regression.fit(x_train,y_train)

#Print the coefficient and intercept
print("Printing regression coeficient: ") # There is coef for every feature
print(regression.coef_)

print("Printing intercept: ") # Intercept is always single value
print(regression.intercept_)

# On which parameter model has been trained
print(regression.get_params())

# Prediction with x_test data
reg_pred=regression.predict(x_test)
print("Regression prediction: ")
print(reg_pred)

#Lets check if prediction is worked well or not by plotting scatter plot
plt.scatter(y_test, reg_pred)
# plt.show() # Plotting should be linear to verify that model works well

#Prediction with residual
residuals=y_test-reg_pred # residuals means error

print(residuals)
#Plot this residual
sns.displot(residuals, kind="kde")
# plt.show() 

# scatter plot wrt prediction and residuals
# plt.scatter(reg_pred, residuals)  # Uniform distribution
# plt.show()

# Using performance metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, reg_pred))
print(mean_squared_error(y_test, reg_pred))

print(np.sqrt(mean_squared_error(y_test, reg_pred)))

# R square and adjusted R square Adjusted R square will be less than R square

from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print("Score", score)

# display adjusted R-squared
adjusted_Rsquare = 1 -(1-score)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)

print("Adjusted Rsquare:", adjusted_Rsquare)

# New data prediction

print(cali.data[0].shape) # This is the shape data has given we need reshape it for regression

print(cali.data[0].reshape(1,-1))
# Need to scale the data before prediction
print(regression.predict(scaler.transform(cali.data[0].reshape(1,-1))))

# Pickling the model file for deployment
pickle.dump(regression, open('regmodel.pkl', 'wb'))

pickled_model=pickle.load(open('regmodel.pkl', 'rb'))

## Prediction 
print(pickled_model.predict(scaler.transform(cali.data[0].reshape(1,-1))))

# Converting entire project to end to end project following industrial standard
