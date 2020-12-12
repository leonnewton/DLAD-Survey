import numpy as np
import matplotlib.pyplot as plt

plt.clf()


layer1_trainloss = np.load("trainlosses_lookback60_layernum1_batch100_epoch15.npy", allow_pickle=True)
layer2_trainloss = np.load("trainlosses_lookback60_layernum2_batch100_epoch15.npy", allow_pickle=True)
layer3_trainloss = np.load("trainlosses_lookback60_layernum3_batch100_epoch15.npy", allow_pickle=True)
layer4_trainloss = np.load("trainlosses_lookback60_layernum4_batch100_epoch15.npy", allow_pickle=True)
layer5_trainloss = np.load("trainlosses_lookback60_layernum5_batch100_epoch15.npy", allow_pickle=True)



plt.plot(layer1_trainloss, label = "1 layer LSTM")
plt.plot(layer2_trainloss, label = "2 layers LSTM")
plt.plot(layer3_trainloss, label = "3 layers LSTM")
plt.plot(layer4_trainloss, label = "4 layers LSTM")
plt.plot(layer5_trainloss, label = "5 layers LSTM")

plt.legend(fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Mean Squared Error (MSE)', fontsize=16)
plt.savefig('./trainlosses.pdf', bbox_inches = "tight")



plt.clf()


layer1_valiloss = np.load("valiloss_lookback60_layernum1_batch100_epoch15.npy", allow_pickle=True)
layer2_valiloss = np.load("valiloss_lookback60_layernum2_batch100_epoch15.npy", allow_pickle=True)
layer3_valiloss = np.load("valiloss_lookback60_layernum3_batch100_epoch15.npy", allow_pickle=True)
layer4_valiloss = np.load("valiloss_lookback60_layernum4_batch100_epoch15.npy", allow_pickle=True)
layer5_valiloss = np.load("valiloss_lookback60_layernum5_batch100_epoch15.npy", allow_pickle=True)

plt.axis([-1,15,0,0.00002])

plt.plot(layer1_valiloss, label = "1 layer LSTM")
plt.plot(layer2_valiloss, label = "2 layers LSTM")
plt.plot(layer3_valiloss, label = "3 layers LSTM")
plt.plot(layer4_valiloss, label = "4 layers LSTM")
plt.plot(layer5_valiloss, label = "5 layers LSTM")

plt.legend(fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Mean Squared Error (MSE)', fontsize=16)
plt.savefig('./valiloss.pdf', bbox_inches = "tight")


plt.clf()


layer1_timeused = np.load("timeused_lookback60_layernum1_batch100_epoch15.npy", allow_pickle=True)
layer2_timeused = np.load("timeused_lookback60_layernum2_batch100_epoch15.npy", allow_pickle=True)
layer3_timeused = np.load("timeused_lookback60_layernum3_batch100_epoch15.npy", allow_pickle=True)
layer4_timeused = np.load("timeused_lookback60_layernum4_batch100_epoch15.npy", allow_pickle=True)
layer5_timeused = np.load("timeused_lookback60_layernum5_batch100_epoch15.npy", allow_pickle=True)



plt.plot(layer1_timeused, label = "1 layer LSTM")
plt.plot(layer2_timeused, label = "2 layers LSTM")
plt.plot(layer3_timeused, label = "3 layers LSTM")
plt.plot(layer4_timeused, label = "4 layers LSTM")
plt.plot(layer5_timeused, label = "5 layers LSTM")

plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0, 1.16), ncol=3)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Time (seconds)', fontsize=16)
plt.savefig('./timeused.pdf', bbox_inches = "tight")




plt.clf()


N = 5
menMeans = (72.9, 169.6, 265.9, 367.6, 464.2)



ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

plt.bar(ind, menMeans, width)
plt.xticks(ind, ('LSTM_1', 'LSTM_2', 'LSTM_3', 'LSTM_4', 'LSTM_5'))
plt.xlabel('Model type', fontsize=16)
plt.ylabel('Model size (KB)', fontsize=16)
plt.savefig('./sizebar.pdf', bbox_inches = "tight")












