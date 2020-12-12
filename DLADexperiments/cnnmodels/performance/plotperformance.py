import numpy as np
import matplotlib.pyplot as plt

plt.clf()


layer1_trainloss = np.load("trainlosses_window_size100_layernum3_batch100_epoch15.npy", allow_pickle=True)
layer2_trainloss = np.load("trainlosses_window_size100_layernum4_batch100_epoch15.npy", allow_pickle=True)
layer3_trainloss = np.load("trainlosses_window_size100_layernum5_batch100_epoch15.npy", allow_pickle=True)
layer4_trainloss = np.load("trainlosses_window_size100_layernum6_batch100_epoch15.npy", allow_pickle=True)
layer5_trainloss = np.load("trainlosses_window_size100_layernum7_batch100_epoch15.npy", allow_pickle=True)



plt.plot(layer1_trainloss, label = "3 layers CNN")
plt.plot(layer2_trainloss, label = "4 layers CNN")
plt.plot(layer3_trainloss, label = "5 layers CNN")
plt.plot(layer4_trainloss, label = "6 layers CNN")
plt.plot(layer5_trainloss, label = "7 layers CNN")

plt.legend(fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Mean Squared Error (MSE)', fontsize=16)
plt.savefig('./cnntrainlosses.pdf', bbox_inches = "tight")



plt.clf()


layer1_valiloss = np.load("valiloss_window_size100_layernum3_batch100_epoch15.npy", allow_pickle=True)
layer2_valiloss = np.load("valiloss_window_size100_layernum4_batch100_epoch15.npy", allow_pickle=True)
layer3_valiloss = np.load("valiloss_window_size100_layernum5_batch100_epoch15.npy", allow_pickle=True)
layer4_valiloss = np.load("valiloss_window_size100_layernum6_batch100_epoch15.npy", allow_pickle=True)
layer5_valiloss = np.load("valiloss_window_size100_layernum7_batch100_epoch15.npy", allow_pickle=True)

# plt.axis([-1,15,0,0.00002])

plt.plot(layer1_valiloss, label = "3 layers CNN")
plt.plot(layer2_valiloss, label = "4 layers CNN")
plt.plot(layer3_valiloss, label = "5 layers CNN")
plt.plot(layer4_valiloss, label = "6 layers CNN")
plt.plot(layer5_valiloss, label = "7 layers CNN")

plt.legend(fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Mean Squared Error (MSE)', fontsize=16)
plt.savefig('./cnnvaliloss.pdf', bbox_inches = "tight")


plt.clf()


layer1_timeused = np.load("timeused_window_size100_layernum3_batch100_epoch15.npy", allow_pickle=True)
layer2_timeused = np.load("timeused_window_size100_layernum4_batch100_epoch15.npy", allow_pickle=True)
layer3_timeused = np.load("timeused_window_size100_layernum5_batch100_epoch15.npy", allow_pickle=True)
layer4_timeused = np.load("timeused_window_size100_layernum6_batch100_epoch15.npy", allow_pickle=True)
layer5_timeused = np.load("timeused_window_size100_layernum7_batch100_epoch15.npy", allow_pickle=True)



layer1_timeused_mean = np.mean(layer1_timeused)
layer2_timeused_mean = np.mean(layer2_timeused)
layer3_timeused_mean = np.mean(layer3_timeused)
layer4_timeused_mean = np.mean(layer4_timeused)
layer5_timeused_mean = np.mean(layer5_timeused)



print(layer1_timeused_mean)
print(layer2_timeused_mean)
print(layer3_timeused_mean)
print(layer4_timeused_mean)
print(layer5_timeused_mean)
# input()

N = 5
menMeans = (layer1_timeused_mean, layer2_timeused_mean, layer3_timeused_mean, layer4_timeused_mean, layer5_timeused_mean)

ind = np.arange(N)    # the x locations for the groups
width = 0.35   

plt.bar(ind, menMeans, width)
plt.xticks(ind, ('3 layers CNN', '4 layers CNN', '5 layers CNN', '6 layers CNN', '7 layers CNN'))
plt.xlabel('Model type', fontsize=16)
plt.ylabel('Epoch time (second)', fontsize=16)
plt.savefig('./cnntimeused.pdf', bbox_inches = "tight")






plt.clf()


N = 5
menMeans = (156.3, 273.4, 395.8, 514.3, 631.3)



ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

plt.bar(ind, menMeans, width)
plt.xticks(ind, ('CNN_3', 'CNN_4', 'CNN_5', 'CNN_6', 'CNN_7'))
plt.xlabel('Model type', fontsize=16)
plt.ylabel('Model size (KB)', fontsize=16)
plt.savefig('./cnnsizebar.pdf', bbox_inches = "tight")












