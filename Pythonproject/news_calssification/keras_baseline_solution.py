
## 파이썬 3에서 한글 텍스트 파일 읽기
f = open('C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/news_calssification/hashtag.txt', 'r', encoding='utf8')
doc_ko = f.read() # 읽어들이면서 지정된 인코딩을 이용하여 문자열 데이터로 변환한다.
print(doc_ko)


## 케라스 시작 
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input
from keras.preprocessing.text import Tokenizer

max_words = 45
batch_size = 32
epochs = 5

from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/news_calssification/val.txt', encoding='utf8', error_bad_lines=False, sep='\t')

## train, val, test data 분할
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df,test_size=0.33, random_state=1000)

# df_train, df_test = np.split(df.sample(frac=1), [int(.8*len(df)), int(.2*len(df))])
df_val = df

## train data 상위 5개 출력
df_train.head()

print("Preparing the Tokenizer...")
tokenizer = Tokenizer(num_words=max_words)

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df_train["text"])

print('Vectorizing sequence data...')

# mode = "binary", "count", "tfidf", "freq" / default = "binary"
x_train = tokenizer.texts_to_matrix(df_train["text"], mode='binary') 
x_test = tokenizer.texts_to_matrix(df_test["text"], mode='binary')
x_val = tokenizer.texts_to_matrix(df_val["text"], mode='binary')

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

x_train

print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')

num_classes = 9

y_train = keras.utils.to_categorical(df_train["class"], num_classes)
y_val = keras.utils.to_categorical(df_val["class"], num_classes)
print('y_train shape:', y_train.shape)
print('y_val shape:', y_val.shape)

y_val

print('Building model sequentially 1...')
model = Sequential()
model.add(Dense(16, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print('Building model sequentially 2...')
model = Sequential([
          Dense(16, input_shape=(max_words,)),
          Activation('relu'),
          Dense(num_classes),
          Activation('softmax')
        ])

model.layers

print(model.to_yaml())

print('Building model functionally...')
a = Input(shape=(max_words,))
b = Dense(16)(a)
b = Activation('relu')(b)
b = Dense(num_classes)(b)
b = Activation('softmax')(b)
model = Model(inputs=a, outputs=b)

from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)

######## plot 실행 에러로 결과값 확인 불가능 : 시작
from keras.utils import plot_model
import pydot
import graphviz

plot_model(model, to_file='model.png', show_shapes=True) ## plot 실행 에러

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
######## plot 실행 에러로 결과값 확인 불가능 : 끝

from keras.objectives import categorical_crossentropy
from keras import backend as K

epsilon = 1.0e-9
def custom_objective(y_true, y_pred):
    '''Yet another crossentropy'''
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    cce = categorical_crossentropy(y_pred, y_true)
    return cce

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.compile(loss=custom_objective,
              optimizer='adam',
              metrics=['accuracy'])

from keras.callbacks import TensorBoard  
tensorboard=TensorBoard(log_dir='./logs', write_graph=True, write_images=True)

from keras.callbacks import EarlyStopping  
early_stopping=EarlyStopping(monitor='val_loss')  


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[tensorboard, early_stopping])

score = model.evaluate(x_val, y_val, batch_size=batch_size, verbose=1)

print('\n')
print('Test score:', score[0])
print('Test accuracy:', score[1])

results = model.predict(x_test, batch_size=batch_size, verbose=1)

prediction = df_test.copy()
prediction["result"] = pd.Series(results[:,1])

prediction[["id","text","class","result"]].to_csv("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/news_calssification/submission.csv", index=False)


