import cv2
import numpy as np

src = cv2.imread('/home/yingkai/Data/T/test11.jpg')
canny = cv2.Canny(src, 174, 255)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
hsv = cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))
(h, w,chanel) = src.shape

canvas = np.zeros((h, w, 1), dtype="uint8")
for i in range(h):
    for j in range(w):
        if canny[i][j]==255 and hsv[i][j]==255:
            canvas[i][j]=255
        else:
            canvas[i][j]=0
cv2.imshow('asdadsa',canvas)
cv2.waitKey()
