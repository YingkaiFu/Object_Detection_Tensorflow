import cv2
import numpy as np

img = cv2.imread('/home/yingkai/Data/data/饼干/item300.jpg')
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hsv = cv2.inRange(hsv,(0,0,0),(180,74,255))
cv2.imshow('result',hsv)
canny = cv2.Canny(hsv,0,150)
cv2.imshow('canny',canny)
cv2.waitKey()
