
# 참고 - https://www.kaggle.com/dcrush/keras-embeddings-with-user-anime-rating

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, merge
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

#from subprocess import check_output
# print(check_output(["ls", "C:/Users/bevis/Downloads/kaggle_anime-recommendations-database"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

anime = pd.read_csv('C:/Users/bevis/Downloads/kaggle_anime-recommendations-database/anime.csv', usecols=['anime_id', 'name'])
users = pd.read_csv('C:/Users/bevis/Downloads/kaggle_anime-recommendations-database/rating.csv')


###---------------
# Read and Prep Data
###---------------

print('Total ratins:', users.shape[0])
print('Excluding unassigned:', (users.rating>-1).sum())

###---------------
# Test Validate split
###---------------

users = users.loc[users.rating>-1, :]
users = users.reindex(np.random.permutation(users.index))
train_index = int(7813737*0.8)
train = users[:train_index]
valid = users[train_index:]
print(train.shape, valid.shape)

train.head()
valid.head()

###---------------
# Create CNN embedding
###---------------

def embedding_inp(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(inp)

user, u = embedding_inp('user_id', users.user_id.nunique(), 20, 1e-4)
anime, m = embedding_inp('anime_id', users.anime_id.nunique(), 20, 1e-4)

x = merge([u, m], mode='concat')
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.7)(x)
x = Dense(1)(x)
cnn = Model([user, anime], x)
callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=0)]
cnn.compile(loss='mse', optimizer="adadelta")

cnn.fit([train.user_id, train.anime_id], train.rating, batch_size=32, nb_epoch=20, 
          validation_data=([valid.user_id, valid.anime_id], valid.rating))