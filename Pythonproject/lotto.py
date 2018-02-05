
import numpy as np

# data
input = pd.read_csv("C:/Users/bevis/Downloads/lotto_785.csv")
data2 = np.array(input)

data = np.array(input)
data = data[:,0:6]

FEATURE_COUNT = 1
buckets = [0] * (FEATURE_COUNT+45)
output = []

for i in reversed(range(1, len(data)+1)):
    week = data[i-1]
    buckets[0] = 786-i  # 회차

    for w in range(0, len(week)):
        idx = week[w]
        buckets[idx] = 1
        if w == len(week)-1:
            output.append(buckets)
            buckets = [0] * (FEATURE_COUNT+45)

print(output)

###-------------
X_train = data2[0:500]
y_train = data2[1:501]

X_test = data2[501:784]
y_test = data2[502:785]

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

input_dim = X_train.shape[1]

model = Sequential()

model.add(Dense(output_dim=45, input_dim=input_dim))
model.add(Activation("relu"))
model.add(Dense(output_dim=7))
model.add(Activation("softmax"))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

epochs = 1000
batch_size = 300

for i in range(100):
    model.fit(X_test, y_test, nb_epoch=epochs, batch_size=batch_size, verbose=0, shuffle=False)
    model.reset_states()

model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch_size)

scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))
print('Test score:', score[0])
print('Test accuracy:', score[1])

predicted = model.predict(X_test, batch_size=batch_size)

##------------------------


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
