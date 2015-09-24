#!/usr/local/bin/python3
import matplotlib.pyplot as plt


f = open('test.txt')
test_datas = f.read().split('\n')
f.close()

x_range = []
input_datas = []
output_datas = []
for i in range(0, len(test_datas) - 1):
    x, input, output = tuple(test_datas[i].split(' '))
    x_range.append(x)
    input_datas.append(input)
    output_datas.append(output)

plt.plot(x_range, input_datas)
plt.plot(x_range, output_datas)
plt.show()


