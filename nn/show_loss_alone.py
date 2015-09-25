#!/usr/local/bin/python3
import matplotlib.pyplot as plt


f = open('loss.txt')
loss = f.read().split('\n')
f.close()

x_range = []
train_datas = []
test_datas = []
for i in range(0, len(loss) - 1):
    x, train, test = tuple(loss[i].split(' '))
    x_range.append(x)
    train_datas.append(train)
    test_datas.append(test)

plt.plot(x_range, train_datas, label='loss_train')
plt.plot(x_range, test_datas, label='loss_test')
plt.legend()
plt.show()


