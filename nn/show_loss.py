#!/usr/local/bin/python3
import matplotlib.pyplot as plt


f = open('loss.txt')
loss = f.read().split('\n')
f.close()
f = open('loss_without_dropout.txt')
loss_without_offset = f.read().split('\n')
f.close()
f = open('loss_with_zerobatch.txt')
loss_noize2 = f.read().split('\n')
f.close()

x_range = []
train_datas = []
test_datas = []
without_offset_train_datas = []
without_offset_test_datas = []
noize2_train_datas = []
noize2_test_datas = []
for i in range(0, len(loss) - 1):
    x, train, test = tuple(loss[i].split(' '))
    x_range.append(x)
    train_datas.append(train)
    test_datas.append(test)
    x, train, test = tuple(loss_without_offset[i].split(' '))
    without_offset_train_datas.append(train)
    without_offset_test_datas.append(test)
    x, train, test = tuple(loss_noize2[i].split(' '))
    noize2_train_datas.append(train)
    noize2_test_datas.append(test)

plt.plot(x_range, train_datas, label='loss_train')
plt.plot(x_range, test_datas, label='loss_test')
plt.plot(x_range, without_offset_train_datas, label='loss_train without dropout')
plt.plot(x_range, without_offset_test_datas, label='loss_test without dropout')
plt.plot(x_range, noize2_train_datas, label='loss_train with zerobatch')
plt.plot(x_range, noize2_test_datas, label='loss_test with zerobatch')
plt.legend()
plt.show()


