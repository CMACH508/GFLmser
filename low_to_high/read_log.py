import numpy as np
from matplotlib import pyplot as plt

file_path = '/home/lipeiying/program/_SR_/Lmser_GAN/Lmser/checkpoint/log/l2.txt'
f = open(file_path, 'r')
lines = f.readlines()

sum_all = 0
cnt = 0
data_list = []
print('file has ', len(lines), 'data')
for i, line in enumerate(lines):
    if line == '' or line == '\n':
        continue
    if i % 10000 == 0:
        # print(line)
        data_list.append(float(line))
        sum_all += float(line)
        cnt += 1

print('choose ', len(data_list), 'data')
x = range(len(data_list))
plt.plot(x, data_list)

plt.xlabel('x')  # X轴标签
plt.ylabel("y")
plt.title("plot")
# plt.savefig("/home/lipeiying/program/_SR_/Lmser_GAN/one.png")
plt.show()
