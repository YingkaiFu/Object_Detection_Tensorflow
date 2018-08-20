import cv2
import numpy as np
pts3 = np.float32([[300, 238], [617, 238], [184, 482], [724, 482]])
pts4 = np.float32([[0, 0], [550, 0], [0, 550], [550, 550]])
M_perspective = cv2.getPerspectiveTransform(pts3, pts4)
image = cv2.imread('img.jpg')
img_perspective = cv2.warpPerspective(image, M_perspective, (550, 550))
cv2.imshow('name',img_perspective)
cv2.waitKey()