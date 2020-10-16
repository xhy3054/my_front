#!/usr/bin/env python3
# encoding=utf-8

'''
展示为图像添加标题、轴标签和图例
'''

import matplotlib.pyplot as plt
import numpy as np

r_gms = np.array([0.338658, 0.693291, 0.764968, 0.811502, 0.875399, 0.923323]) 
p_gms = np.array([0.981481, 0.943478, 0.945088, 0.916968, 0.878205, 0.58502])
r_dwms = np.array([0.37, 0.430064, 0.572716, 0.722173, 0.757188, 0.85623, 0.960466])
p_dwms = np.array([1.0, 0.990973, 0.987143, 0.957143, 0.955645, 0.899329, 0.787547])
r_ransac = np.array([0.380192, 0.488818, 0.565495, 0.637588, 0.7795553, 0.858101, 0.929712])
p_ransac = np.array([0.929688, 0.864407, 0.859223, 0.849785, 0.849155, 0.84623, 0.679907])
r_ratio = np.array([0.341853, 0.479233, 0.603834, 0.769968, 0.955272]) 
p_ratio = np.array([0.856, 0.78125, 0.598101, 0.444649, 0.330752])

#图标题
plt.title("Tum")
#轴标签
plt.xlabel("recall")
plt.ylabel("precision")

# 此处X与Y是matplotlib.lines.Line2D object类型
X, = plt.plot(r_gms, p_gms, "r--o")
# plt.plot()获得的是一个只有一个元素的矩阵
Y, = plt.plot(r_dwms, p_dwms, "b--^")
# 此处之所以要用 X,= 而不是 X= 是因为后者会使得X是一个矩阵类型，这个矩阵中的那个元素才是我们想要的
Z, = plt.plot(r_ransac, p_ransac, "g--v")

S, = plt.plot(r_ratio, p_ratio, "y--*")
plt.xticks(np.arange(0.3,1.0,0.1))

#图例
plt.legend([X,Y,Z,S], ["GMS", "DWMS", "RANSAC", "ratio-test"])


plt.show()
