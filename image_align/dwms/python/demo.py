import cv2 as cv
import numpy as np
import argparse
from flann.dwms import dwms_matcher
import need.other as func

## [load]
parser = argparse.ArgumentParser(description='Code for AKAZE local features matching tutorial.')
parser.add_argument('--input1', help='Path to input image 1.', default='boat/img1.pgm')
parser.add_argument('--input2', help='Path to input image 2.', default='boat/img4.pgm')
parser.add_argument('--homography', help='Path to the homography matrix.', default='boat/H1to4p')
args = parser.parse_args()

img1 = cv.imread(cv.samples.findFile(args.input1))
img2 = cv.imread(cv.samples.findFile(args.input2))
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)

h = func.read_homography(cv.samples.findFile(args.homography))
## [load]
################################# orb begin #####################################################
## [orb]
n = 1000
orb = cv.ORB_create(n)
orb.setFastThreshold(0);
kp1, d1 = orb.detectAndCompute(img1, None)
kp2, d2 = orb.detectAndCompute(img2, None)
## [orb]

## [orb matching]
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
matches_orb = matcher.match(d1, d2)
## [orb matching]

h_max_error = 2.5
orb_test = func.compute_inliers(kp1, kp2, matches_orb, h, h_max_error)
n_orb = len(matches_orb)
n_orb_true = 0
for x in orb_test:
    if x==True:
        n_orb_true += 1
res_orb = func.draw_result(img1, img2, kp1, kp2, matches_orb, orb_test)
cv.imwrite("orb_result.png", res_orb)


print('ORB Matching Results')
print('*******************************')
print('# Matches:                            \t', n_orb)
print('# inliers:                            \t', n_orb_true)
print('# precision:                          \t', n_orb_true/n_orb)

################################## orb end ####################################################

################################# dwms begin ##################################################
dwms = dwms_matcher(kp1, kp2, matches_orb, 20, 1.0)
dwms_res = dwms.GetInlier()
n_dwms=0
n_dwms_true=0
matches_dwms = []
dwms_test = []
for i, test in enumerate(dwms_res):
    if test==True:
        n_dwms += 1
        matches_dwms.append(matches_orb[i])   
        if orb_test[i]==True:
            n_dwms_true +=1
            dwms_test.append(True)
        else:
            dwms_test.append(False)

res_dwms = func.draw_result(img1, img2, kp1, kp2, matches_dwms, dwms_test)
cv.imwrite("dwms_result.png", res_dwms)

print('\n \n')
print('dwms Matching Results')
print('*******************************')
print('# Matches:                            \t', n_dwms)
print('# inliers:                            \t', n_dwms_true)
print('# precision:                          \t', n_dwms_true/n_dwms)
print('# recall:                             \t', n_dwms_true/n_orb_true)

################################# dwms end ####################################################

cv.imshow('result_orb', res_orb)
cv.imshow('result_dwms', res_dwms)
cv.waitKey()
## [draw final matches]