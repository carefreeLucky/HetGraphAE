import test as g
import matplotlib.pyplot as plt

loss = g.loss
fp = open('testData/anomaly_index.txt')
index = eval(fp.read())
fp.close()

# 异常进程的异常分数集
loss_anomy = loss[index]
loss_anomy.sort()

count_p = 0
count_tp = 0

# 做出进程与其对应异常分数的散点图
key = list(range(len(loss)))
value = loss
label = ['y'] * len(key)
for i in range(len(key)):
    if i in index:
        label[i] = 'r'
plt.scatter(y=value, x=key, s=0.1, c=label)
plt.title('the anomaly score of process')
plt.show()

p_sum = len(loss_anomy)
n_sum = len(loss) - len(loss_anomy)


def f(idx):
    thres = loss_anomy[idx]
    p = 0
    tp = 0
    for i in range(len(loss)):
        if i in index:
            if loss[i] >= thres:
                tp += 1
        if loss[i] >= thres:
            p += 1
    recall = tp / len(loss_anomy)
    precision = tp / p
    f1 = (2 * recall * precision) / (recall + precision)
    return recall, precision, f1


recall = []
precison = []
f1 = []
for i in range(len(loss_anomy)):
    r, p, F = f(i)
    recall.append(r)
    precison.append(p)
    f1.append(F)

ln1, = plt.plot(recall, precison, color='red', linewidth=2.0, linestyle='--')
ln2, = plt.plot(recall, f1, color='blue', linewidth=2.0, linestyle='-.')
plt.legend(handles=[ln1, ln2], labels=['precision', 'f1'])
plt.xlabel('recall')
plt.ylabel('value')
plt.show()
