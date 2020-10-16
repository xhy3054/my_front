#!/usr/bin/env python3
# encoding=utf-8

'''
展示为图像添加标题、轴标签和图例
'''

import matplotlib.pyplot as plt
import numpy as np

x = np.array([2,3,4,5,6]) 
y1 = np.array([88.8376,77.5551,23.7113,53.8462,0])
y2 = np.array([92.9761,93.401,85.3026,78.6207,48.1928])

#图标题
plt.title("F")
#轴标签
#plt.xlabel("X")
#plt.ylabel("Y")

# 此处X与Y是matplotlib.lines.Line2D object类型
X, = plt.plot(x, y1, "r--o")
# plt.plot()获得的是一个只有一个元素的矩阵
Y, = plt.plot(x, y2, "b--*")
# 此处之所以要用 X,= 而不是 X= 是因为后者会使得X是一个矩阵类型，这个矩阵中的那个元素才是我们想要的

plt.xticks(np.arange(2,7,1))

#图例
plt.legend([X,Y], ["GMS", "DWMS"])


plt.show()
