from tensorflow.python.keras.layers import Input, Dense, LSTM, Conv2D, MaxPooling2D, UpSampling2D
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
import seaborn as sns



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



def plotvalilosses(originline, name):
    plt.switch_backend('agg')
    plt.clf()
    plt.plot(originline)
    plt.savefig('./images/valiloss_' + name + '.png')









epochnum = 15
batchsize = 100
window_size = 100
step = 10
layernum = 7



storename = "window_size"+str(window_size)+"_layernum"+str(layernum)+"_batch"+str(batchsize) + "_epoch" +str(epochnum)
#-------------cnn model---------------------
input_img = Input(shape=(20, 20, 1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)


x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')


print(autoencoder.summary())
# input()
#-------------end cnn model---------------------


sensordata = "../DLADexperiment/data/normalsensors.npy"
X = np.load(sensordata)[:,0:20]



start = 41000
end =   43000

plotnormal = X[start:end, 8].reshape((-1,1))
print("plotnormal: ", plotnormal.shape)

plt.clf()
plt.plot(plotnormal)
plt.savefig('./images/normal.png')

# input()

#normalize the data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)


sensor_n = X.shape[1]
matrix_all = []

for t in range(0, X.shape[0]-window_size, step):
	matrix_t = np.zeros((sensor_n, sensor_n))
	for i in range(sensor_n):
		for j in range(i, sensor_n):
			#if np.var(raw_data[i, t - win:t]) and np.var(raw_data[j, t - win:t]):
			matrix_t[i][j] = np.inner(X[t:t+window_size,i], X[t:t+window_size,j])/(window_size) # rescale by win
			matrix_t[j][i] = matrix_t[i][j]

	
	matrix_all.append(matrix_t)

matrix_all = np.array(matrix_all)
matrix_all = np.reshape(matrix_all, (len(matrix_all), 20, 20, 1))
print("matrix all: ", matrix_all.shape, len(matrix_all))

#split into training and testing set
train_size = int(len(matrix_all) * 0.8)
test_size = len(matrix_all) - train_size
train, test = matrix_all[0:train_size,:,:,:], matrix_all[train_size:len(matrix_all),:,:,:]

print("train shape: ", train.shape, "test shape: ", test.shape)


history1 = LossHistory()
timerecord = TimeHistory()

historyresult = autoencoder.fit(train, train, epochs=epochnum, batch_size=batchsize, verbose=2, callbacks=[history1, timerecord], validation_data=(test, test))


valiloss = np.array(historyresult.history['val_loss'])
print("training loss mean: ", np.mean(history1.losses), history1.losses[-10:])
plotlosses(history1.losses, storename)
plotvalilosses(valiloss, storename)


trainlosses = np.array(history1.losses)
np.save("./performance/trainlosses_"+storename+".npy", trainlosses)
np.save("./performance/valiloss_"+storename+".npy", valiloss)


timeused = timerecord.times
timeused = np.array(timeused)
np.save("./performance/timeused_"+storename+".npy", timeused)


testcase = matrix_all[200,:,:,:].reshape((1, 20, 20, 1))
# print(testcase.shape)

reconstructcase = autoencoder.predict(testcase)
# print(reconstructcase.shape)

testcase = testcase.reshape((20, 20))
reconstructcase = reconstructcase.reshape((20, 20))

plt.clf()
ax = sns.heatmap(testcase, vmin=0, vmax=1.5)
figure = ax.get_figure()    
figure.savefig('./images/testcase.png', dpi=400)

plt.clf()
ax = sns.heatmap(reconstructcase, vmin=0, vmax=1.5)
figure = ax.get_figure()    
figure.savefig('./images/reconstructcase.png', dpi=400)

plt.clf()
diff = np.absolute(reconstructcase - testcase)
ax = sns.heatmap(diff, vmin=0, vmax=0.4, cbar_kws={'label': 'Reconstruction errors'})
ax.set_xlabel('Sensors', fontname="Arial",  fontsize=16)
ax.set_ylabel('Sensors', fontname="Arial",  fontsize=16)
ax.set_title('Reconstruction errors of Matrix M', y=-0.2)

figure = ax.get_figure()    
figure.tight_layout()
figure.savefig('./images/diff.png', dpi=400)
# input()


autoencoder.save("./performance/cnn_model_" + storename + ".h5")


#---------------------start test attack data-----------------

attacksensor = "../DLADexperiment/data/attacksensors.npy"
X = np.load(attacksensor, allow_pickle=True)[:,0:20]

start = 7000
end =   9000

plotattack = X[start:end,8].reshape((-1,1))
plt.clf()
plt.xlabel('Time (s)', fontname="Arial",  fontsize=18)
plt.ylabel('Water level sensor values (mm)', fontname="Arial",  fontsize=18)
plt.plot(plotattack)
plt.savefig('./images/attack.pdf')


X = sc.transform(X)



start = 7000
end = 9000

X = X[start:end,:]
print("attack sensors: ", X.shape)



sensor_n = X.shape[1]
matrix_all = []

for t in range(0, X.shape[0]-window_size, step):
	matrix_t = np.zeros((sensor_n, sensor_n))
	for i in range(sensor_n):
		for j in range(i, sensor_n):
			#if np.var(raw_data[i, t - win:t]) and np.var(raw_data[j, t - win:t]):
			matrix_t[i][j] = np.inner(X[t:t+window_size,i], X[t:t+window_size,j])/(window_size) # rescale by win
			matrix_t[j][i] = matrix_t[i][j]

	
	matrix_all.append(matrix_t)

matrix_all = np.array(matrix_all)
matrix_all = np.reshape(matrix_all, (len(matrix_all), 20, 20, 1))
print("matrix all: ", matrix_all.shape, len(matrix_all))

for i in range(len(matrix_all)):
	testcase = matrix_all[i,:,:,:].reshape((1, 20, 20, 1))
	reconstructcase = autoencoder.predict(testcase)

	testcase = testcase.reshape((20, 20))
	reconstructcase = reconstructcase.reshape((20, 20))

	plt.clf()
	ax = sns.heatmap(testcase)
	figure = ax.get_figure()    
	figure.savefig('./images/attack_testcase'+str(i)+'.png', dpi=400)

	plt.clf()
	ax = sns.heatmap(reconstructcase)
	figure = ax.get_figure()    
	figure.savefig('./images/attack_reconstructcase'+str(i)+'.png', dpi=400)

	plt.clf()
	diff = np.absolute(reconstructcase - testcase)
	ax = sns.heatmap(diff, vmin=0, vmax=0.4, cbar_kws={'label': 'Reconstruction errors'})
	ax.set_xlabel('Sensors', fontname="Arial",  fontsize=16)
	ax.set_ylabel('Sensors', fontname="Arial",  fontsize=16)
	ax.set_title('Reconstruction errors of Matrix M', y=-0.2)
	figure = ax.get_figure()    
	figure.tight_layout()
	figure.savefig('./images/attack_diff'+str(i)+'.png', dpi=400)










