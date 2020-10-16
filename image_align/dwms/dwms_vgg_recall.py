#!/usr/bin/env python3
# encoding=utf-8

'''
展示为图像添加标题、轴标签和图例
'''

import matplotlib.pyplot as plt
import numpy as np

x = np.array([2,3,4,5,6]) 
y1 = np.array([88.906,68.984,13.7725,38.8889,0])
y2 = np.array([98.9214,98.3957,88.6229,73.0769,33.8983])

#图标题
plt.title("Recall")
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
