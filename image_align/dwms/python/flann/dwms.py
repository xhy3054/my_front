
'dwms_matcher module'

__author__ = 'xhy'

import cv2 as cv
import numpy as np
import math
'''
    FLANN_INDEX_LINEAR = 0,
    FLANN_INDEX_KDTREE = 1,
    FLANN_INDEX_KMEANS = 2,
    FLANN_INDEX_COMPOSITE = 3,
    FLANN_INDEX_KDTREE_SINGLE = 4,
    FLANN_INDEX_HIERARCHICAL = 5,
    FLANN_INDEX_LSH = 6,
    FLANN_INDEX_SAVED = 254,
    FLANN_INDEX_AUTOTUNED = 255, 
'''
class dwms_matcher(object):
    """docstring for dwms_matcher"""
    def __init__(self, kp1, kp2, matches_orb, k=20, threshold=1.2):
        self.mp1 = []
        self.mp2 = []
        self.convert_point(kp1, kp2)

        params = dict(algorithm=1, trees=1)
        self.flann1 = cv.flann_Index(np.float32(self.mp1), params)
        self.flann2 = cv.flann_Index(np.float32(self.mp2), params)

        self.matches_map = {}
        self.matches_orb = matches_orb
        self.convert_match()

        self.k = k
        self.thres = threshold


    def convert_point(self, kp1, kp2):
        for keypoint in kp1:
            self.mp1.append(keypoint.pt)
        for keypoint in kp2:
            self.mp2.append(keypoint.pt)

    def convert_match(self):
        for match in self.matches_orb:
            self.matches_map[match.queryIdx] = match.trainIdx

    def GetInlier(self):
        dwms_test = []
        for match in self.matches_orb:
            if self.check_match(match) == True:
                #print('a true match')
                dwms_test.append(True)
            else:
                dwms_test.append(False)
        return dwms_test
        


    def check_match(self, match):
        #first find the k neighbors, get the indexs
        search_mat1 = np.float32(self.mp1[match.queryIdx])
        indices1, dists1 = self.flann1.knnSearch(search_mat1, self.k, -1)
        search_mat2 = np.float32(self.mp2[match.trainIdx])
        indices2, dists2 = self.flann2.knnSearch(search_mat2, int(self.k*1.25), -1)

        sum = 0
        for i in indices1[0]:
            if self.matches_map[i] in indices2[0]:
                sum += 1
        threshold = self.thres * math.sqrt(1.25*self.k)
        if sum > threshold:
            return True
        else:
            return False

        
