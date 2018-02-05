## 케라스 실습
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer

df = pd.read_csv('C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/news_calssification/val.txt', encoding='utf8', error_bad_lines=False, sep='\t')

## train, val, test data 분할
df_train, df_val, df_test = np.split(df.sample(frac=1), [int(.5*len(df)), int(.5*len(df))])

## train data 상위 5개 출력
df_train.head()

max_words = 45
batch_size = 32
epochs = 5

print("Preparing the Tokenizer...")
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_train["text"])

print('Vectorizing sequence data...')

x_train = tokenizer.texts_to_matrix(df_train["text"], mode='binary')
x_test = tokenizer.texts_to_matrix(df_test["text"], mode='binary')
x_val = tokenizer.texts_to_matrix(df_val["text"], mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

x_train

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')

num_classes = 16

y_train = keras.utils.to_categorical(df_train["class"], num_classes)
y_test = keras.utils.to_categorical(df_test["class"], num_classes)
y_val = keras.utils.to_categorical(df_val["class"], num_classes)
print('y_train shape:', y_train.shape)
print('y_val shape:', y_val.shape)

y_val

# cross-validation 할때 같은 번호들이 선택되지 않고 골고루 선택되기 위해 shuffling 함 

shuffle_index = np.random.permutation(23247)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(x_train, y_train_5):
    clone_clf = clone(sgd_clf)
    x_train_folds = x_train[train_index]
    y_train_folds = (y_train_5[train_index])
    x_test_fold = x_train[test_index]
    y_test_fold = (y_train_5[test_index])

    # test set 을 모델에 넣어 prediction 값을 꺼내고 맞는 비율을 확인
    clone_clf.fit(x_train_folds, y_train_folds)
    y_pred = clone_clf.predict(x_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))