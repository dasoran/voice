#!/usr/local/bin/python3
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

f = open('test.txt')
test_datas = f.read().split('\n')
f.close()

input_datas = []
output_datas = []

for i in range(0, len(test_datas) - 1):
    x, input, output_ = tuple(test_datas[i].split(' '))
    output_data4append = math.floor(float(output_))
    input_data4append = math.floor(float(input))
    if output_data4append < 0:
        output_data4append = 0
    output_datas.append(output_data4append)
    input_datas.append(input_data4append)


output(result_export_path, output_datas)
output(input_export_path, input_datas)


