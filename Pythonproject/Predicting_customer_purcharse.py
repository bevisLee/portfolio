### --------------------------------------------------
### --- Lezhin Customer Purchase Prediction 
### --- chlee
### --- 16 Aug 2017.
### --- Online Purchase Data Set
### --------------------------------------------------

## 참조 - http://blog.yhat.com/posts/predicting-customer-churn-with-sklearn.html



## Data set import
from __future__ import division
import pandas as pd
import numpy as np

churn_df = pd.DataFrame
churn_df = pd.read_csv('C:/Users/bevis/Downloads/lezhin_dataset_v2_training.tsv/lezhin_dataset_v2_training.tsv', sep='\t')
col_names = churn_df.columns.tolist()

print "Column names:"
print col_names

to_show = col_names[:]

print "\nSample data:"
churn_df[to_show].head(6)

# Isolate target data --> 여기부터 시작
churn_result = churn_df['pay_YN']
y = np.where(churn_result == 'True.',1,0)

# We don't need these columns
to_drop = ['State','Area Code','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1)

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

# Pull out features for future use
features = churn_feat_space.columns

X = churn_feat_space.as_matrix().astype(np.float)

# This is important
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print "Feature space holds %d observations and %d features" % X.shape
print "Unique target labels:", np.unique(y)
      
