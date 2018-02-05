

# 참조 - http://3months.tistory.com/168

## load data set
from pandas_datareader import data
import fix_yahoo_finance as yf

yf.pdr_override()

start_date = '1996-05-06' #startdate를 1996년으로 설정해두면 가장 오래된 데이터부터 전부 가져올 수 있다.
tickers = ['067160.KQ', '035420.KS'] #1 아프리카tv와 네이버의 ticker(종목코드)
# afreeca = data.get_data_yahoo(tickers[0], start_date)
naver = data.get_data_yahoo(tickers[1], start_date)

## 1
# 주목해야 할 점은 아프리카tv는 코스닥 종목이고, 네이버는 코스피 종목이라는 것. 코스닥은 ticker에 `.KQ`, 코스피는 `.KS`가 붙는다.
## 2
# data객체의 get_data_yahoo 를 이용해서 얻은 데이터들(아프리카, 네이버)은 DataFrame형의 객체이다. (정확히는 pandas.core.frame.DataFrame) data객체가 pandas-datareader에서 가져온 것임을 잊지말자

naver.head()
naver.to_csv('C:/Users/bevis/Downloads/tf_RNN_lstm/naver.csv')

naver = naver[["Close"]]

# 최근 데이터로 재추출
naver2 = naver.loc['2013-01-01':, ['Close']]
naver2.to_csv('C:/Users/bevis/Downloads/tf_RNN_lstm/naver2.csv')

import matplotlib.pyplot as plt
import pandas as pd

train_date = pd.Timestamp('2015-06-20')

train = naver2.loc[:train_date, ['Close']]
test = naver2.loc[train_date:, ['Close']]

ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'val','test'])
plt.show()

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()

train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

train_sc.shape
train_sc.head()

train_sc_df = pd.DataFrame(train_sc, columns=['Scaled'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Scaled'], index=test.index)
train_sc_df.head()

for s in range(1, 13):
    train_sc_df['shift_{}'.format(s)] = train_sc_df['Scaled'].shift(s)
    test_sc_df['shift_{}'.format(s)] = test_sc_df['Scaled'].shift(s)

train_sc_df.head(13)

X_train = train_sc_df.dropna().drop('Scaled', axis=1)
y_train = train_sc_df.dropna()[['Scaled']]

X_test = test_sc_df.dropna().drop('Scaled', axis=1)
y_test = test_sc_df.dropna()[['Scaled']]

X_train.head()
y_train.head()

X_train = X_train.values
X_test= X_test.values

y_train = y_train.values
y_test = y_test.values

print(X_train.shape)
print(X_train)
print(y_train.shape)
print(y_train)

X_train_t = X_train.reshape(X_train.shape[0], 12, 1) 
X_test_t = X_test.reshape(X_test.shape[0], 12, 1)

print("최종 DATA")
print(X_train_t.shape)
print(X_train_t)
print(y_train.shape)
print(y_train)

## LSTM Model
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

import keras.backend as K 
from keras.callbacks import EarlyStopping 

# 모델 구성하기
K.clear_session() 
model = Sequential() # Sequeatial Model 
model.add(LSTM(20, input_shape=(12, 1))) # (timestep, feature 
model.add(Dense(1)) # output = 1 
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

# model fitting
for i in range(100):
    model.fit(X_train_t, y_train, epochs=100, batch_size=30, verbose=1, callbacks=[early_stop])
    model.reset_states()

print(X_test_t)
y_pred = model.predict(X_test_t)
print(y_pred)

## plot
plt.plot(y_test)
plt.plot(y_pred)
plt.legend(['y_test', 'y_pred'])
plt.show()

# inverse transform : real data
Y_test_predict = sc.inverse_transform(y_test)
Y_predict = sc.inverse_transform(y_pred)

## plot
plt.plot(Y_test_predict)
plt.plot(Y_predict)
plt.legend(['Y_test_predict', 'Y_predict'])
plt.show()



