import cv2 as cv
import numpy as np
from math import sqrt
from math import pow

# 读入单应矩阵
def read_homography(s):
    temp = []
    with open(s) as f:
        temp = f.read().split()
    H1toN = np.zeros((3,3), dtype=np.float64)
    H1toN[0,0] = float(temp[0]) / float(temp[8])
    H1toN[0,1] = float(temp[1]) / float(temp[8])
    H1toN[0,2] = float(temp[2]) / float(temp[8])
    H1toN[1,0] = float(temp[3]) / float(temp[8])
    H1toN[1,1] = float(temp[4]) / float(temp[8])
    H1toN[1,2] = float(temp[5]) / float(temp[8])
    H1toN[2,0] = float(temp[6]) / float(temp[8])
    H1toN[2,1] = float(temp[7]) / float(temp[8])
    H1toN[2,2] = float(temp[8]) / float(temp[8])
    return H1toN

# 判断匹配是否正确
def compute_inliers(kp1, kp2, Dmatches, H, h_max_error):
    result=[]
    h11 = H[0,0]
    h12 = H[0,1]
    h13 = H[0,2]
    h21 = H[1,0]
    h22 = H[1,1]
    h23 = H[1,2]
    h31 = H[2,0]
    h32 = H[2,1]
    h33 = H[2,2]
    for match in Dmatches:
        x1 = kp1[match.queryIdx].pt[0]
        y1 = kp1[match.queryIdx].pt[1]
        x2 = kp2[match.trainIdx].pt[0]
        y2 = kp2[match.trainIdx].pt[1]        

        s = h31*x1 + h32*y1 + h33
        x2m = (h11*x1 + h12*y1 + h13)/s
        y2m = (h21*x1 + h22*y1 + h23)/s
        dist = sqrt(pow(x2m-x2, 2) + pow(y2m-y2, 2))
        
        if dist <= h_max_error:
            result.append(True)
        else:
            result.append(False)

    return result

# 画出匹配结果
def draw_result(img1, img2, kp1, kp2, matches, matches_test):
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    res = np.empty((height, width, 3),  dtype=np.uint8)
    res[: , 0:img1.shape[1]] = img1.copy()
    res[: , img1.shape[1]:width] = img2.copy()

    for i, match in enumerate(matches):
        left = kp1[match.queryIdx].pt
        right = (kp2[match.trainIdx].pt[0]+img1.shape[1], kp2[match.trainIdx].pt[1])
        if matches_test[i] == True:
            cv.line(res, (int(left[0]), int(left[1])), (int(right[0]), int(right[1])), (0,255,255))
        else:
            cv.line(res, (int(left[0]), int(left[1])), (int(right[0]), int(right[1])), (0,0,255))
            #cv.line(res, left, right, (0,0, 255))

    return res


if __name__=='__main__':
    h = read_homography('H1to2p')
    print(h)
