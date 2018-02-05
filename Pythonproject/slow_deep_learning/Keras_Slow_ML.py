## 1. 케라스 DNN
# 출처 : https://www.kaggle.com/dakshmiglani/credit-card-fraudulent-detection-with-dnn-keras/notebook
# Credit Card Fraudulent Detection

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras

df = pd.read_csv('C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/slow_deep_learning/creditcard.csv')
df.head(1)

df['Class'].unique() # 0 = no fraud, 1 = fraudulent

# 데이터와 라벨 분리
# 트레이닝 셋과 테스트셋 분리

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# 데이터 정규화

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout # Dense 모두 연결해서 활용 / # Dropout 랜덤하게 죽여서 사용

# 네트워크 설계

clf = Sequential([
    Dense(units=16, kernel_initializer='uniform', input_dim=30, activation='relu'),
    Dense(units=18, kernel_initializer='uniform', activation='relu'),
    Dropout(0.25),
    Dense(20, kernel_initializer='uniform', activation='relu'),
    Dense(24, kernel_initializer='uniform', activation='relu'),
    Dense(1, kernel_initializer='uniform', activation='sigmoid') # 마지막에 2진 분류
])

clf.summary()

# DNN 학습

clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # 15개씨 loss 계산해서 2바퀴 진행

clf.fit(X_train, Y_train, batch_size=15, epochs=2)

score = clf.evaluate(X_test, Y_test, verbose=0)
print(clf.metrics_names)
print(score)

## 2. 케라스 CNN

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, pooling # 20*20 풀때 - Flatten, Conv2D, pooling
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

# 데이터 준비
# 데이터 차원
# Backend로 Theano를 사용할 경우엔 (channel, width, height)
# Tensorflow를 사용할 경우엔 (width, height, channel)

(X_train, y_train),(X_test, y_test) = mnist.load_data() #다운로드
len(X_train) 

# 인풋을 (데이터개수,28 ,28,1)이 되도록 리쉐이프
X_train = X_train.reshape(X_train.shape[0], 28, 28,1) # Theano 1,28,28
X_test = X_test.reshape(X_test.shape[0], 28, 28,1) # Theano 1,28,28

# 데이터 타입 변환 및 노멀라이즈
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 클래스(라벨)을 범주형 데이터로 변환
print(y_train.shape)
print(y_train[:10])
n_class=10 # n_class : 클래스 갯수 정의

Y_train = np_utils.to_categorical(y_train, n_class) 
Y_test = np_utils.to_categorical(y_test, n_class) 
print(Y_train[:10])

## CNN 네트워크 생성
model = Sequential() #CNN 이나 RNN은 시퀀셜

#컨볼루션 레이어
model.add(Convolution2D(32, 3, 3, activation='relu',input_shape=(28,28,1)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
          
#풀리커넥티트 레이어
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#학습알고리즘 정의 및 학습
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1)

#테스트
score = model.evaluate(X_test, Y_test, verbose=0)
print(model.metrics_names)
print(score)

## 3. 케라스 RNN
# 시계열을 입력으로 다음 스텝의 출력을 예측하는 단순한 시계열 예측 문제
# 데이터
# 사인파형 시계열 데이터
## 참고 사이트 - https://datascienceschool.net/view-notebook/1d93b9dc6c624fbaa6af2ce9290e2479/

import matplotlib.pyplot as plt

s = np.sin(2 * np.pi * 0.125 * np.arange(20))
plt.plot(s, 'ro-')
plt.xlim(-0.5, 20.5)
plt.ylim(-1.1, 1.1)
plt.show()

# Keras 에서 RNN의 입력 데이터는 (nb_samples, timesteps, input_dim) 3차원 텐서(tensor) 형태임
 # nb_samples: 자료의 수
 # timesteps: 순서열의 길이
 # input_dim: x 벡터의 크기
# 단일 시계열이므로 input_dim=1 이고 3 스텝 크기의 순서열을 사용하므로 timesteps=3 이며 자료의 수는 18개

from scipy.linalg import toeplitz
S = np.fliplr(toeplitz(np.r_[s[-1], np.zeros(s.shape[0] - 2)], s[::-1]))
S[:5, :3]

X_train = S[:-1, :3][:, :, np.newaxis]
Y_train = S[:-1, 3]
X_train.shape, Y_train.shape

X_train[:2]

Y_train[:2]

plt.subplot(211)
plt.plot([0, 1, 2], X_train[0].flatten(), 'bo-', label="input sequence")
plt.plot([3], Y_train[0], 'ro', label="target")
plt.xlim(-0.5, 4.5)
plt.ylim(-1.1, 1.1)
plt.legend()
plt.title("First sample sequence")
plt.subplot(212)
plt.plot([1, 2, 3], X_train[1].flatten(), 'bo-', label="input sequence")
plt.plot([4], Y_train[1], 'ro', label="target")
plt.xlim(-0.5, 4.5)
plt.ylim(-1.1, 1.1)
plt.legend()
plt.title("Second sample sequence")
plt.tight_layout()
plt.show()

## 케라스 rnn 만드는 순서
# Sequential 클래스 객체인 모형을 생성한다.
# add 메서드로 다양한 레이어를 추가한다.
# compile 메서드로 목적함수 및 최적화 방법을 지정한다.
# fit 메서드로 가중치를 계산한다.

import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

np.random.seed(0)
model = Sequential()
model.add(SimpleRNN(10, input_dim=1, input_length=3))
model.add(Dense(1))
model.compile(loss='mse', optimizer='sgd')

plt.plot(Y_train, 'ro-', label="target")
plt.plot(model.predict(X_train[:,:,:]), 'bs-', label="output")
plt.xlim(-0.5, 20.5)
plt.ylim(-1.1, 1.1)
plt.legend()
plt.title("Before training")
plt.show()

#학습
history = model.fit(X_train, Y_train, nb_epoch=100, verbose=0)

plt.plot(history.history["loss"])
plt.title("Loss")
plt.show()

plt.plot(Y_train, 'ro-', label="target")
plt.plot(model.predict(X_train[:,:,:]), 'bs-', label="output")
plt.xlim(-0.5, 20.5)
plt.ylim(-1.1, 1.1)
plt.legend()
plt.title("After training")
plt.show()

