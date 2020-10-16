#!/usr/bin/env python3
# encoding=utf-8

'''
展示为图像添加标题、轴标签和图例
'''

import matplotlib.pyplot as plt
import numpy as np


#定义一个实例来容纳所有子图
fig = plt.figure()
#subplot函数的三个参数分别是行、列、活跃区域编号
ax1 = fig.add_subplot(1,3,1)    
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)    


x = np.array([2,3,4,5,6]) 

#第一个子图
ax1.set_title("Precison")
gms_precision = np.array([0.887692,0.885583,0.851852,0.875,0.24])
dwms_precision = np.array([0.877049,0.888889,0.822222,0.850746,0.697674])
X, = ax1.plot(x, gms_precision, "r-o")
Y, = ax1.plot(x, dwms_precision, "g-^")
ax1.set_xticks(np.arange(2,7,1))
ax1.set_yticks(np.arange(0,1.1,0.1))
ax1.legend([X,Y], ["GMS", "DWMS"])


#第二个子图
ax2.set_title("Recall")
gms_recall = np.array([0.88906,0.68984,0.137725,0.388889,0])
dwms_recall = np.array([0.989214,0.983957,0.886229,0.730769,0.338983])
X, = ax2.plot(x, gms_recall, "r-o")
Y, = ax2.plot(x, dwms_recall, "g-^")
ax2.set_xticks(np.arange(2,7,1))
ax2.set_yticks(np.arange(0,1.1,0.1))
ax2.legend([X,Y], ["GMS", "DWMS"])


#第三个子图
ax3.set_title("F1")
gms_f1 = np.array([0.888376,0.775551,0.237113,0.538462,0])
dwms_f1 = np.array([0.929761,0.93401,0.853026,0.786207,0.481928])
X, = ax3.plot(x, gms_f1, "r-o")
Y, = ax3.plot(x, dwms_f1, "g-^")
ax3.set_xticks(np.arange(2,7,1))
ax3.set_yticks(np.arange(0,1.1,0.1))
ax3.legend([X,Y], ["GMS", "DWMS"])



plt.show()
