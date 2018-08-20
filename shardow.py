import cv2
import numpy as np


pts3 = np.float32([[256, 225], [600, 230], [111, 501], [734, 496]])
# pts3 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
pts4 = np.float32([[0, 0], [550, 0], [0, 550], [550, 550]])
M_perspective = cv2.getPerspectiveTransform(pts3, pts4)

image = cv2.imread('/home/yingkai/Data/8.9/test1.jpg')
cv2.imshow('original',image)
img_perspective = cv2.warpPerspective(image, M_perspective, (550, 550))
cv2.imshow('perspect',img_perspective)
#img_perspective = cv2.GaussianBlur(img_perspective,(3,3),0)
f_edges = cv2.Canny(img_perspective, 80 , 31, apertureSize=3)
cv2.imshow('f_edge', f_edges)
cv2.waitKey()