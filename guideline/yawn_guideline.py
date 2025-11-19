# -----------------------------------------------------------------------------
# Face Tracking and Evaluation Algorithm
# -----------------------------------------------------------------------------
# Author: Daniel Oliveira
# https://github.com/danielsousaoliveira
# -----------------------------------------------------------------------------
#출처:https://github.com/danielsousaoliveira/driving-monitor-python/blob/main/detection/face.py

import cv2
import numpy as np
from numpy import linalg as LA
import time

#MAR(Mouth Aspect Ratio) 계산/ MAR이 하품을 계산하는데 중요한 요소, MAR이 임계값이 넘으면 하품으로 측정
def calculate_mouth_aspect_ratio(self, upperLips, lowerLips):

        """ Calculate the mouth aspect ratio.

        Args:
            upperLips (numpy.ndarray): Upper lip landmark points.
            lowerLips (numpy.ndarray): Lower lip landmark points.

        Returns:
            float: The mouth aspect ratio.
        """
        # LA.norm은 유클리드 거리를 계산합니다.
        marAvg = (LA.norm(upperLips[14] - lowerLips[17]) # 수직 거리 1
                   + LA.norm(upperLips[12] - lowerLips[14])) / (LA.norm(upperLips[0] - upperLips[8]) # 수평 거리 1
                                                                  + LA.norm(lowerLips[12] - lowerLips[10])) # 수평 거리 2

        return marAvg       
