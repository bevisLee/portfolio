

## 0. 데이터 파악 
import pandas as pd

rcdata = pd.read_csv('C:/Users/bevis/Downloads/work_rcdata/rcData_v7.csv')

rcdata.info()

rcdata.describe()

rcdata.head()

rcdata.shape

import matplotlib.pyplot as plt
rcdata.hist(bins=50, figsize=(20,15))
plt.show()


## test------------
rcdata['rcResult_3'] = rcdata['rcResult_3'].astype('category')
rcdata['hrSex'] = rcdata['hrSex'].astype('category')
rcdata['rcType'] = rcdata['rcType'].astype('category')
rcdata['rcClass'] = rcdata['rcClass'].astype('category')
rcdata['track'] = rcdata['track'].astype('category')
rcdata['weather'] = rcdata['weather'].astype('category')
rcdata['trRcCntT'] = rcdata['trRcCntT'].astype('category')

rcdata["rcResult_3"] = rcdata["rcResult_3"].cat.codes
rcdata["hrSex"] = rcdata["hrSex"].cat.codes
rcdata["rcType"] = rcdata["rcType"].cat.codes
rcdata["rcClass"] = rcdata["rcClass"].cat.codes
rcdata["track"] = rcdata["track"].cat.codes
rcdata["weather"] = rcdata["weather"].cat.codes
rcdata["trRcCntT"] = rcdata["trRcCntT"].cat.codes

rcdata_log = rcdata

y = rcdata.rcResult_3 # define the target variable (dependent variable) as y

X_train, X_test, y_train, y_test = train_test_split(rcdata, y, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

##-------
import statsmodels.api as sm
import statsmodels.formula.api as smf

lm1 = smf.ols(formula='rcResult_3 ~.', data=rcdata).fit()

# print the coefficients
lm1.params

##-------

## 1. Linear Regression

#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model

#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays

x_train=input_variables_values_training_datasets
y_train=target_variables_values_training_datasets
x_test=input_variables_values_test_datasets

# Create linear regression object
linear = linear_model.LinearRegression()

# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)

#Equation coefficient and Intercept

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

#Predict Output
predicted= linear.predict(x_test)