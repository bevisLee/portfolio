
## 참고 - https://github.com/NourozR/Stock-Price-Prediction-LSTM

## load data set
from pandas_datareader import data
import fix_yahoo_finance as yf

yf.pdr_override()

start_date = '1996-05-06' #startdate를 1996년으로 설정해두면 가장 오래된 데이터부터 전부 가져올 수 있다.
tickers = ['067160.KQ', '035420.KS'] #1 아프리카tv와 네이버의 ticker(종목코드)
# afreeca = data.get_data_yahoo(tickers[0], start_date)
naver = data.get_data_yahoo(tickers[1], start_date)

naver.head()
# naver.to_csv('C:/Users/bevis/Downloads/tf_RNN_lstm/naver.csv')

naver = naver[["Open", "High", "Low", "Close", "Volume"]]
naver2 = naver.loc['2013-01-01':, ["Open", "High", "Low", "Close", "Volume"]]

## Preprocessing
import numpy as np 

# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)

# IMPORTING IMPORTANT LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.callbacks import EarlyStopping 

# FOR REPRODUCIBILITY
np.random.seed(7)

## IMPORTING DATASET 
# dataset = pd.read_csv('C:/Users/bevis/Downloads/tf_RNN_lstm/apple_share_price.csv', usecols=[1,2,3,4])
# dataset = dataset.reindex(index = dataset.index[::-1]) # 내림차순일 경우만 사용
dataset = naver2

# CREATING OWN INDEX FOR FLEXIBILITY
obs = np.arange(1, len(dataset) + 1, 1)

# TAKING DIFFERENT INDICATORS FOR PREDICTION
OHLC_avg = dataset.mean(axis = 1)
HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
close_val = dataset[['Close']]

"""
# PLOTTING ALL INDICATORS IN ONE PLOT
plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
plt.plot(obs, HLC_avg, 'b', label = 'HLC avg')
plt.plot(obs, close_val, 'g', label = 'Closing price')
plt.legend(loc = 'upper left')
plt.show()
"""

# PREPARATION OF TIME SERIES DATASE
OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)

# TRAIN-TEST SPLIT
train_OHLC = int(len(OHLC_avg) * 0.75)
test_OHLC = len(OHLC_avg) - train_OHLC
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = new_dataset(train_OHLC, 1)
testX, testY = new_dataset(test_OHLC, 1)

# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1

# LSTM MODEL1
model1 = Sequential()
model1.add(LSTM(32, input_shape=(1,step_size), return_sequences = True))
model1.add(LSTM(16))
model1.add(Dense(1))
model1.add(Activation('linear'))

# MODEL COMPILING AND TRAINING
model1.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad and compare!!!

# LSTM MODEL2
model2 = Sequential()
model2.add(LSTM(32, input_shape=(1,step_size), return_sequences = True))
model2.add(LSTM(16))
model2.add(Dense(1))
model2.add(Activation('linear'))

# MODEL COMPILING AND TRAINING
model2.compile(loss='mean_squared_error', optimizer='SGD') # Try SGD, adam, adagrad and compare!!!

# LSTM MODEL3
model3 = Sequential()
model3.add(LSTM(32, input_shape=(1,step_size), return_sequences = True))
model3.add(LSTM(16))
model3.add(Dense(1))
model3.add(Activation('linear'))

# MODEL COMPILING AND TRAINING
model3.compile(loss='mean_squared_error', optimizer='adam') # Try SGD, adam, adagrad and compare!!!

early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

# model fitting
for i in range(10):
    print("model1 fitting : adagrad & ",i," fitting")
    model1.summary()
    model1.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[early_stop])
    model1.reset_states()
    print("model2 fitting : SGD & ",i," fitting")
    model2.summary()
    model2.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[early_stop])
    model2.reset_states()
    print("model3 fitting : adam & ",i," fitting")
    model3.summary()
    model3.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, callbacks=[early_stop])
    model3.reset_states()

# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) # 기존

# PREDICTION
trainPredict1 = model1.predict(trainX)
trainPredict2 = model2.predict(trainX)
trainPredict3 = model3.predict(trainX)

testPredict1 = model1.predict(testX)
testPredict2 = model2.predict(testX)
testPredict3 = model3.predict(testX)

# DE-NORMALIZING FOR PLOTTING
trainPredict1 = scaler.inverse_transform(trainPredict1)
trainPredict2 = scaler.inverse_transform(trainPredict2)
trainPredict3 = scaler.inverse_transform(trainPredict3)
trainY = scaler.inverse_transform([trainY])

testPredict1 = scaler.inverse_transform(testPredict1)
testPredict2 = scaler.inverse_transform(testPredict2)
testPredict3 = scaler.inverse_transform(testPredict3)
testY = scaler.inverse_transform([testY])

# TRAINING RMSE
trainScore1 = math.sqrt(mean_squared_error(trainY[0], trainPredict1[:,0]))
trainScore2 = math.sqrt(mean_squared_error(trainY[0], trainPredict2[:,0]))
trainScore3 = math.sqrt(mean_squared_error(trainY[0], trainPredict3[:,0]))
trainScore = min(trainScore1,trainScore2,trainScore3)

print('Train1 RMSE: %.2f' % (trainScore1))
print('Train2 RMSE: %.2f' % (trainScore2))
print('Train3 RMSE: %.2f' % (trainScore3))
print('Train RMSE: %.2f' % (trainScore))

# TEST RMSE
testScore1 = math.sqrt(mean_squared_error(testY[0], testPredict1[:,0]))
testScore2 = math.sqrt(mean_squared_error(testY[0], testPredict2[:,0]))
testScore3 = math.sqrt(mean_squared_error(testY[0], testPredict3[:,0]))
testScore = min(testScore1,testScore2,testScore3)

print('Test1 RMSE: %.2f' % (testScore1))
print('Test2 RMSE: %.2f' % (testScore2))
print('Test3 RMSE: %.2f' % (testScore3))
print('Test RMSE: %.2f' % (testScore))

trainPredict = trainPredict1
testPredict = testPredict1

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(OHLC_avg)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(OHLC_avg)
testPredictPlot[:, :] = np.nan
# testPredictPlot[len(trainPredict)+(step_size*2):len(OHLC_avg)-2, :] = testPredict
testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict # 기존

# DE-NORMALIZING MAIN DATASET 
OHLC_avg = scaler.inverse_transform(OHLC_avg)

# PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
plt.plot(OHLC_avg, 'b', label = 'original dataset')
plt.plot(trainPredictPlot, 'g', label = 'training set')
plt.plot(testPredictPlot, 'r', label = 'predicted stock price/test set')
plt.legend(loc = 'upper left')
plt.xlabel('Time in Days')
plt.ylabel('Stocks')
plt.show()

# PREDICT FUTURE VALUES
last_val = testPredict[-1]
last_val_scaled = last_val/last_val
next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
print ("Last Day Value:", np.asscalar(last_val))
print ("Next Day Value:", np.asscalar(last_val*next_val))
# print np.append(last_val, next_val)

