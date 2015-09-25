import numpy as np
#import matplotlib.pyplot as plt
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import wave
import struct
import math
import os

# define default constance
#default_bitrate = 48000
default_bitrate = 1000
datas_export_path = '/home/dasoran/datas.wav'
sample_output = 'test.txt'
loss_output = 'loss.txt'


# generate data
def input(path):
    w = wave.open(path, mode='rb')
    i = 0
    length = w.getnframes()
    raw_data = w.readframes(length)
    datas = list(struct.unpack('={0:d}H'.format(length), raw_data))
    w.close()
    return datas

def output(path, datas, bitrate = default_bitrate):
    output_datas = bytearray()
    length = len(datas)
    i = 0
    while i < length:
        byted_data = datas[i].to_bytes(2,  byteorder='little')
        output_datas.append(byted_data[0])
        output_datas.append(byted_data[1])
        i = i + 1
    w = wave.open(path, mode='wb')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(bitrate)
    w.writeframes(output_datas)
    w.close()

datas = []
n_all_batch = 0


if os.path.exists(datas_export_path):
    file_path = datas_export_path
    file_datas = input(file_path)
    n_this_batch = math.floor(len(file_datas) / default_bitrate)
    use_datas = file_datas[0:(default_bitrate * n_this_batch)]
    datas.extend(use_datas)
    n_all_batch = n_all_batch + n_this_batch
    print('loaded: existed data'.format(id))
else:
    for id in range(1, 100):
        file_path = '/Users/dasoran/voice/world_wav/File{0:04d}.ogg.wav'.format(id)
        file_datas = input(file_path)
        n_this_batch = math.floor(len(file_datas) / default_bitrate)
        use_datas = file_datas[0:(default_bitrate * n_this_batch)]
        datas.extend(use_datas)
        n_all_batch = n_all_batch + n_this_batch
        print('loaded: world_wav/File{0:04d}.ogg.wav'.format(id))
    output(datas_export_path, datas)

print(n_all_batch)



# define constance
batchsize = 1000
n_input = default_bitrate
n_units = 500
n_epoch = 20


# remove zero
zero_removed_datas = []
n_removed_batch = 0
n_ok_batch = 0
for i in range(0, n_all_batch):
    one_batch_datas = datas[i*default_bitrate:(i+1)*default_bitrate]
    #isAllZero = True
    isAllZero = False
    for data in one_batch_datas:
        if data > 10:
            #isAllZero = False
            break
    if isAllZero:
        n_removed_batch = n_removed_batch + 1
    else:
        n_ok_batch = n_ok_batch + 1
        zero_removed_datas.extend(one_batch_datas)

n_all_batch = n_all_batch - n_removed_batch
print(n_ok_batch, n_removed_batch)
print(n_all_batch, len(zero_removed_datas) / default_bitrate)

# make noizied copy
noizied_copy_datas = []
for i in range(0, n_all_batch):
    one_batch_datas = np.array(zero_removed_datas[i*default_bitrate:(i+1)*default_bitrate])
    rand_noizes = np.random.rand(default_bitrate) * 1000 - 500
    one_batch_datas = (one_batch_datas + rand_noizes).tolist()
    noizied_copy_datas.extend(one_batch_datas)
zero_removed_datas.extend(noizied_copy_datas)
n_all_batch = n_all_batch * 2

print(n_all_batch, len(zero_removed_datas) / default_bitrate)

#np_datas = np.array(datas, dtype=np.float32) - 30000
#np_datas = np.array(zero_removed_datas, dtype=np.float32)
np_datas = np.array(zero_removed_datas, dtype=np.float32)
batched_datas = np_datas.reshape((n_all_batch, default_bitrate))
batched_datas = batched_datas.astype(np.float32)
#n_train_batchset = n_all_batch - 50000
n_train_batchset = n_all_batch - 222960
x_test, x_train = np.split(batched_datas, [n_all_batch - n_train_batchset])
y_test, y_train = np.split(batched_datas.copy(), [n_all_batch - n_train_batchset])
n_test_batchset = math.floor(x_test.size / default_bitrate)

print(x_train.size / default_bitrate, x_train.ndim)
print(x_test.size / default_bitrate, x_test.ndim)
print(n_train_batchset, n_test_batchset, n_train_batchset + n_test_batchset, n_all_batch)


model = FunctionSet(
    l1 = F.Linear(n_input, n_units),
    l2 = F.Linear(n_units, n_input)
).to_gpu()


def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)), train=train)
    y = F.dropout(model.l2(h1), train=train)
    return F.mean_squared_error(y, t)


optimizer = optimizers.Adam()
optimizer.setup(model)


loss_train = []
loss_test = []
for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(n_train_batchset)
    sum_loss = 0
    for i in range(0, n_train_batchset, batchsize):
        x_batch = cuda.to_gpu(x_train[perm[i:i + batchsize]])
        y_batch = cuda.to_gpu(y_train[perm[i:i + batchsize]])

        optimizer.zero_grads()
        #loss = forward(x_batch, y_batch)
        loss = forward(x_batch, y_batch, train=True)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(x_batch)

    print('train mean loss={}'.format(
        sum_loss / n_train_batchset))
    loss_train.append(sum_loss / n_train_batchset)

    # evaluation
    sum_loss = 0
    for i in range(0, n_test_batchset, batchsize):
        x_batch = cuda.to_gpu(x_test[i:i + batchsize])
        y_batch = cuda.to_gpu(y_test[i:i + batchsize])

        loss = forward(x_batch, y_batch, train=False)

        sum_loss += float(loss.data) * len(x_batch)

    print('test  mean loss={}'.format(
        sum_loss / n_test_batchset))
    loss_test.append(sum_loss / n_test_batchset)


# output loss
f = open(loss_output, 'w')
for i in range(1, n_epoch + 1):
    strs = '{0:05.0f} {1:.2f} {2:.2f}\n'.format(i, loss_train[i - 1], loss_test[i - 1])
    f.writelines(strs)
f.close()


# test
print('-----')
print('starting to make test data with model')

x = Variable(cuda.to_gpu(x_test[1000:1000 + 200].reshape((200, default_bitrate))))
h1 = F.dropout(F.relu(model.l1(x)),  train=False)
y = F.dropout(model.l2(h1), train=False)
#print(x.data)
#print(y.data)


x_range = np.arange(0, default_bitrate * 200, 1)
print('test  mean loss={}'.format(F.mean_squared_error(y, x).data))
#print(x.data.ndim, x_range.ndim)
#plt.plot(x_range, y.data[0])
#plt.plot(x_range, t.data[0])
#plt.show()
x_datas = []
t_datas = []
y_datas = []
x_datas.extend(x_range)
for t_in_onebatch in x.data.tolist():
    t_datas.extend(t_in_onebatch)
for y_in_onebatch in y.data.tolist():
    y_datas.extend(y_in_onebatch)

print('finished converting to save')

strs = ""
for i in range(0, len(x_datas)):
    strs = strs + '{0:05.0f} {1:.2f} {2:.2f}\n'.format(x_datas[i], t_datas[i], y_datas[i])
print('finished making outputdata')
f = open(sample_output, 'w')
f.write(strs)
f.close()


