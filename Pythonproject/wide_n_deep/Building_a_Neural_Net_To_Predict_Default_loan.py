## Introduction
# The intention of this notebook is to utilize tensorflow to build a neural network that helps to predict default likelihood, and to visualize some of the insights generated from the study. This kernel will evolve over time as I continue to add features and study the Lending Club data

# Dependencies

# Below the data and some external libraries are imported to begin the process
#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import itertools
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat

tf.logging.set_verbosity(tf.logging.DEBUG) # FATAL, WARN, INFO, DEBUG

df = pd.read_csv("C:/Users/bevis/Downloads/lending-club-loan-data/loan.csv", low_memory=False,encoding='CP949') # Default_Binary, Purpose_Cat 사전 작업

df = pd.read_csv("C:/Users/bevis/Downloads/lending-club-loan-data/loan2.csv", low_memory=False,encoding='CP949')
## Creating the Target Label

# From a prior notebook, I examined the 'loan_status' column. The cell below creates a column with binary value 0 for loans not in default, and binary value 1 for loans in default.

####-------다른 선언하는 방법 확인 후 수정 필요 : 시작
df['Default_Binary'] = int(0)

def f(row):
    if row['loan_status'] == 'Default' :
        val = int(1)
    elif row['loan_status'] == 'Charged Off' :
        val = int(1)
    elif row['loan_status'] == 'Late (31-120 days)' :
        val = int(1)
    elif row['loan_status'] == 'Late (16-30 days)' :
        val = int(1)
    elif row['loan_status'] == 'Does not meet the credit policy. Status:Charged Off' :
        val = int(1)
    else:
        val = 0
    return val

df['Default_Binary'] = df.apply(f, axis=1)

## Creating a category feature for "Loan Purpose"

# Below I create a new column for loan purpose, and assign each type of loan purpose an integer value.

df['Purpose_Cat'] = int(0) 

def e(row):
    if row['purpose'] == 'debt_consolidation' :
        val = int(1)
    elif row['purpose'] == 'credit_card' :
        val = int(2)
    elif row['purpose'] == 'home_improvement' :
        val = int(3)
    elif row['purpose'] == 'other' :
        val = int(4)
    elif row['purpose'] == 'major_purchase' :
        val = int(5)
    elif row['purpose'] == 'small_business' :
        val = int(6)
    elif row['purpose'] == 'car' :
        val = int(7)
    elif row['purpose'] == 'medical' :
        val = int(8)
    elif row['purpose'] == 'moving' :
        val = int(9)
    elif row['purpose'] == 'vacation' :
        val = int(10)
    elif row['purpose'] == 'house' :
        val = int(11)
    elif row['purpose'] == 'wedding' :
        val = int(12)
    elif row['purpose'] == 'renewable_energy' :
        val = int(13)
    elif row['purpose'] == 'educational' :
        val = int(14)
    else:
        val = 0
    return val

df['Purpose_Cat'] = df.apply(e, axis=1)

####-------다른 선언하는 방법 확인 후 수정 필요 : 끝

# Now I use get_dummies to create new features

# I also create the frame that will be used in the net

df_train = pd.get_dummies(df.purpose).astype(int)

df_train.columns = ['debt_consolidation','credit_card','home_improvement',
                     'other','major_purchase','small_business','car','medical',
                     'moving','vacation','house','wedding','renewable_energy','educational']

# Also add the target column we created at first
df_train['Default_Binary'] = df['Default_Binary']
df_train['Purpose_Cat'] = df['Purpose_Cat']
df_train.head()

## Scaling Interest Rates

# Below I scale the interest rate for each loan to a value between 0 and 1

x = np.array(df.int_rate.values).reshape(-1,1) 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['int_rate_scaled'] = pd.DataFrame(x_scaled)
print (df.int_rate_scaled[0:5])

## Add i rate and loan amount to the df_train frame

x = np.array(df.funded_amnt.values).reshape(-1,1) 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['funded_amnt_scaled'] = pd.DataFrame(x_scaled)
print (df.funded_amnt_scaled[0:5])

df_train['int_rate_scaled'] = df['int_rate_scaled']
df_train['funded_amnt_scaled'] = df['funded_amnt_scaled'] # 122~126 선언 추가

## Setting up the Neural Network

# Below I split the data into a training, testing, and prediction set

# After that, I assign the feature and target columns, and create the function that will be used to pass the data into the model

# Cell below is under construction to divide the input fn between continuous and categorical data

# Note for future, need to add an estimator function for features
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/wide_n_deep_tutorial.py

training_set = df_train[0:500000] # Train on first 500k rows
testing_set = df_train[500001:800000] # Test on next 400k rows
prediction_set = df_train[800001:] # Predict on final ~87k rows

COLUMNS = ['debt_consolidation','credit_card','home_improvement',
           'other','major_purchase','small_business','car','medical',
           'moving','vacation','house','wedding','renewable_energy','educational',
           'funded_amnt_scaled','int_rate_scaled','Default_Binary']   

FEATURES = ['debt_consolidation','credit_card','home_improvement',
           'other','major_purchase','small_business','car','medical',
           'moving','vacation','house','wedding','renewable_energy','educational',
           'funded_amnt_scaled','int_rate_scaled'] 

#CONTINUOUS_COLUMNS = ['funded_amnt_scaled','int_rate_scaled'] 
#CATEGORICAL_COLUMNS = ['Purpose_Cat']

LABEL = 'Default_Binary'

def input_fn(data_set):
    ### Simple Version ######
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES} # Working method for continous data DO NOT DELETE 
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels
    
    """
     # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(data_set[k].values)
                     for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(data_set[k].size)],
        values=data_set[k].values,
        shape=[data_set[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    #feature_cols = dict(continuous_cols.items() + categorical_cols.items()) # Throws error
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels
    """

## Fitting The Model

learning_rate = 0.01
feature_cols = [tf.contrib.layers.real_valued_column(k)
              for k in FEATURES]
#config = tf.contrib.learn.RunConfig(keep_checkpoint_max=1) ######## DO NOT DELETE
regressor = tf.contrib.learn.DNNRegressor(
                    feature_columns=feature_cols,
                    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
                    hidden_units=[10, 20, 10], )

regressor.fit(input_fn=lambda: input_fn(training_set), steps=500)

## Evaluating the Model

# Score accuracy
ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

## Predicting on new data
y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
predictions = list(itertools.islice(y, 87378))

## Visualize Predictions Relative To Interest Rates
plt.plot(prediction_set.int_rate_scaled, predictions, 'ro')
plt.ylabel("Model Prediction Value")
plt.xlabel("Interest Rate of Loan (Scaled between 0-1)")
plt.show()

## Visualize Predictions Relative to Loan Size
plt.plot(prediction_set.funded_amnt_scaled, predictions, 'ro')
plt.ylabel("Model Prediction Value")
plt.xlabel("Funded Amount of Loan (Scaled between 0-1)")
plt.show()

## Visualize Predictions Relative to Loan Purpose
plt.plot(prediction_set.Purpose_Cat, predictions, 'ro')
plt.ylabel("Model Prediction Value")
plt.xlabel("Loan Purpose")
plt.title("DNN Regressor Predicting Default By Loan Purpose")
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
labels = ['Debt Consolidation', 'Credit Card', 'Home Improvement',            'Other','Major Purchase', 'Small Business', 'Car', 
          'Medical','Moving', 'Vacation', 'House', 'Wedding',
          'Renewable Energy','educational']

plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14], labels, rotation='vertical')

plt.show()



