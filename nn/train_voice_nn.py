import numpy as np
import matplotlib.pyplot as plt
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import wave
import struct
import math
import os

# define default constance
default_bitrate = 48000


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

datas_export_path = '/Users/dasoran/datas.wav'

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
batchsize = 100
n_input = default_bitrate
n_units = 1000
n_epoch = 20


np_datas = np.array(datas, dtype=np.float32)
batched_datas = np_datas.reshape((n_all_batch, default_bitrate))
batched_datas = batched_datas.astype(np.float32)
n_train_batchset = math.floor(n_all_batch / batchsize) - 1
x_train, x_test = np.split(batched_datas,   [n_train_batchset * batchsize])
n_test_batchset = math.floor(x_test.size / default_bitrate)

print(x_train.size / default_bitrate, x_train.ndim)
print(x_test.size / default_bitrate, x_test.ndim)


model = FunctionSet(
    l1 = F.Linear(n_input, n_units),
    l2 = F.Linear(n_units, n_units),
    l3 = F.Linear(n_units, n_input)
)


def forward(x_data, train=True):
    x = Variable(x_data)
    h1 = F.dropout(F.relu(model.l1(x)),  train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)
    return F.mean_squared_error(x, y)


optimizer = optimizers.Adam()
optimizer.setup(model)


for epoch in range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(n_train_batchset)
    sum_loss = 0
    for i in range(0, n_train_batchset, batchsize):
        x_batch = np.asarray(x_train[perm[i:i + batchsize]])

        optimizer.zero_grads()
        loss = forward(x_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(x_batch)

    print('train mean loss={}'.format(
        sum_loss / n_train_batchset))

    # evaluation
    sum_loss = 0
    for i in range(0, n_test_batchset, batchsize):
        x_batch = np.asarray(x_test[i:i + batchsize])

        loss = forward(x_test, train=False)

        sum_loss += float(loss.data) * len(x_test)

    print('test  mean loss={}'.format(
        sum_loss / n_test_batchset))

# test
print(x_train[3])
x = Variable(x_train[3].reshape((1, default_bitrate)))
h1 = F.dropout(F.relu(model.l1(x)),  train=False)
h2 = F.dropout(F.relu(model.l2(h1)), train=False)
y = model.l3(h2)

x_range = np.arange(0, default_bitrate, 1)
print(F.mean_squared_error(x, y).data)
print(x.data.ndim, x_range.ndim)
plt.plot(x_range, x.data[0])
plt.plot(x_range, y.data[0])
plt.show()

