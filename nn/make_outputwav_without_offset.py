#!/usr/local/bin/python3
import numpy as np
import wave
import math

default_bitrate = 48000
result_export_path = '/Users/dasoran/ml/voice/nn/result.wav'
input_export_path = '/Users/dasoran/ml/voice/nn/input.wav'


def output(path, datas, bitrate = default_bitrate):
    output_datas = bytearray()
    length = len(datas)
    i = 0
    while i < length:
        byted_data = math.floor(datas[i]).to_bytes(2,  byteorder='little')
        output_datas.append(byted_data[0])
        output_datas.append(byted_data[1])
        i = i + 1
    w = wave.open(path, mode='wb')
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(bitrate)
    w.writeframes(output_datas)
    w.close()

f = open('test.txt')
test_datas = f.read().split('\n')
f.close()

input_datas = []
output_datas = []

n_cutting_top_output = 0
n_cutting_bottom_output = 0
n_cutting_top_input = 0
n_cutting_bottom_input = 0
for i in range(0, len(test_datas) - 1):
    x, input, output_ = tuple(test_datas[i].split(' '))
    output_data4append = math.floor(float(output_)) + 2000
    if output_data4append > 65535:
        output_data4append = 65535
        n_cutting_top_output = n_cutting_top_output + 1
    if output_data4append < 0:
        output_data4append = 0
        n_cutting_bottom_output = n_cutting_bottom_output + 1
    input_data4append = math.floor(float(input))
    if input_data4append > 65535:
        input_data4append = 65535
        n_cutting_top_input = n_cutting_top_input + 1
    if input_data4append < 0:
        input_data4append = 0
        n_cutting_bottom_input = n_cutting_bottom_input + 1
    output_datas.append(output_data4append)
    input_datas.append(input_data4append)

print('cutting top in OUTPUT     : {}'.format(n_cutting_top_output))
print('cutting bottom in OUTPUT  : {}'.format(n_cutting_bottom_output))
print('cutting top in INPUT      : {}'.format(n_cutting_top_input))
print('cutting bottom in INPUT   : {}'.format(n_cutting_bottom_input))

output(result_export_path, (np.array(output_datas)).tolist())
output(input_export_path, (np.array(input_datas)).tolist())


