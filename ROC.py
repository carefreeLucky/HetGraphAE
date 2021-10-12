import matplotlib.pyplot as plt
import numpy as np

loss = np.load('HetGraphAELoss.npy')
fp = open('testData/anomaly_index.txt')
index = eval(fp.read())
fp.close()

loss_anomy = loss[index]
loss_anomy.sort()

p_sum = len(loss_anomy)
n_sum = len(loss) - len(loss_anomy)


def f(thres):
    p = 0
    tp = 0
    for i in range(len(loss)):
        if i in index:
            if loss[i] >= thres:
                tp += 1
        if loss[i] >= thres:
            p += 1
    return tp / p_sum, (p - tp) / n_sum


TPR = []
FPR = []
i = 0
f_to_t = dict()
loss_set = list(set(loss))
loss_set.sort()
for i in range(len(loss_set)):
    tpr, fpr = f(loss_set[i])
    TPR.append(tpr)
    FPR.append(fpr)
FPR.append(0)
TPR.append(0)
TPR.reverse()
FPR.reverse()
AUC = 0
pre_x = 0
for i in range(0, len(FPR)):
    if FPR[i] != pre_x:
        AUC += (FPR[i] - pre_x) * TPR[i]
        pre_x = FPR[i]
print("AUC", AUC)
plt.plot(FPR, TPR, label='HetGraphAE')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.title('The ROC curves')
plt.show()
