from tensorflow.python.keras.layers import Input, Dense, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import metrics
import numpy.linalg as nplin
from collections import Counter
from scipy.stats import kstest
from sklearn.metrics import mean_squared_error
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import Sequential
import time




class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)






def plotlosses(line, name):
    plt.switch_backend('agg')
    plt.clf()
    plt.plot(line)
    plt.savefig('./images/loss_' + name + '.png')

def plotattack(line, name):
    plt.switch_backend('agg')
    plt.clf()
    plt.plot(line)
    plt.savefig('./images/attack_' + name + '.png')



def plottimehistory(line, name):
    plt.switch_backend('agg')
    plt.clf()
    plt.plot(line)
    plt.savefig('./images/time_' + name +'.png')


def plotoriginsensor(originline, name):
    plt.switch_backend('agg')
    plt.clf()
    plt.plot(originline)
    plt.savefig('./images/origin_' + name + '.png')


def plotvalilosses(originline, name):
    plt.switch_backend('agg')
    plt.clf()
    plt.plot(originline)
    plt.savefig('./images/valiloss_' + name + '.png')


def plotpredictsensor(originline, name):
    plt.switch_backend('agg')
    plt.clf()
    plt.plot(originline)
    plt.savefig('./images/predict_' + name + '.png')


def create_traindata(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)


#hyperparameters
look_back = 60
layernum = 5
batchsize = 100
epochnum = 15

storename = "lookback"+str(look_back)+"_layernum"+str(layernum)+"_batch"+str(batchsize) + "_epoch" +str(epochnum)

# sensor data here
sensordata = "../DLADexperiment/data/normalsensorlit101.npy"
sensorname ="lit101"


X = np.load(sensordata)


X = X.reshape((-1,1))
print("X shape: ", X.shape, len(X))




#normalize the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)


#split the data into training set and testing set
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
train, test = X[0:train_size,:], X[train_size:len(X),:]


train_X, train_y = create_traindata(train, look_back)
print("train_X, train_y: ", train_X.shape, train_y.shape)

test_X, test_y = create_traindata(test, look_back)
print("test_X, test_y: ", test_X.shape, test_y.shape)


#build the model
model = Sequential()
model.add(LSTM(30, input_shape=(look_back, 1), return_sequences=True))
model.add(LSTM(30, return_sequences=True))
model.add(LSTM(30, return_sequences=True))
model.add(LSTM(30, return_sequences=True))
model.add(LSTM(30))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


history1 = LossHistory()
timerecord = TimeHistory()
historyresult = model.fit(train_X, train_y, epochs=epochnum, batch_size=batchsize, verbose=2, callbacks=[history1, timerecord], validation_data=(test_X, test_y))
valiloss = np.array(historyresult.history['val_loss'])


print("training loss mean: ", np.mean(history1.losses), history1.losses[-10:])


print(model.summary())

trainlosses = np.array(history1.losses)
np.save("./performanceevaluation/trainlosses_"+storename+".npy", trainlosses)
np.save("./performanceevaluation/valiloss_"+storename+".npy", valiloss)


plotlosses(history1.losses, storename)
plotvalilosses(valiloss, storename)

timeused = timerecord.times
timeused = np.array(timeused)
np.save("./performanceevaluation/timeused_"+storename+".npy", timeused)
plottimehistory(timeused, storename)


plot_model(model, show_shapes=True, to_file='./images/lstm_model_'+storename+'.png')



model.save("./performanceevaluation/lstm_model_" + storename + ".h5")



#-----------------------------------------------------------
plotstart = 8000
plotend = 10000

normalplotline = test_X[plotstart:plotend,:,:]
realplotline = test_y[plotstart:plotend]

yhat = model.predict(normalplotline)
print("yhat: ", yhat.shape)


np.save("./plotdata/realplotline.npy", realplotline)
np.save("./plotdata/yhat.npy", yhat)


plt.switch_backend('agg')
plt.clf()
xrow = np.arange(yhat.shape[0])
plt.plot(xrow, realplotline)
plt.plot(xrow, yhat)
plt.savefig('./images/originandpredict_' + storename + '.png')


#---------------------------------------------------------

attackdata = "../DLADexperiment/data/attacksensorlit101.npy"
X = np.load(attackdata, allow_pickle=True)

X = X.reshape((-1,1))
print("X shape: ", X.shape, len(X))


X = sc.transform(X)




start = 4000
end = 6000
attack1 = X[start:end]
print("attack1 shape: ", attack1.shape, len(attack1))




attack1_name = "1"
plotattack(attack1, attack1_name)


attack_X, attack_y = create_traindata(attack1, look_back)

attacky_hat = model.predict(attack_X)

print(attacky_hat.shape, attack_y.shape)

diff = attack1.shape[0] - attacky_hat.shape[0]

plt.switch_backend('agg')
plt.clf()
xrow = np.arange(attacky_hat.shape[0])
plt.plot(xrow, attack1[diff:])
plt.plot(xrow, attacky_hat)
plt.savefig('./images/attack_predict_' + storename + '.png')

detectionwindow = 60
attackoriginal = attack1[diff:]

differnce = np.absolute(attacky_hat - attackoriginal)




np.save("./differnce.npy", differnce)




