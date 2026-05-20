import numpy as np
import cv2
import json
import scipy.io 


# .mat으로 읽기
mat = scipy.io.loadmat('camparams.mat')
K = mat['K']
dist = mat['distCoeffs'].reshape(-1,)

# 이미지 읽기
img = cv2.imread('image_to_undistort.jpg')  # 원본 이미지 파일명
h, w = img.shape[:2]

# 출력을 원본 영역(full) 또는 same
# 1) 간단히 undistort
undistorted = cv2.undistort(img, K, dist, None, K)

# 2) 더 좋은 결과: Optimal new camera matrix 사용
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)  # alpha=0: 잘린다, 1: 모든 픽 유지
undistorted2 = cv2.undistort(img, K, dist, None, newK)


# 결과 저장
cv2.imwrite('undistorted.jpg', undistorted2)