import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


src = cv2.imread('/home/yingkai/Pictures/Screenshot from 2018-07-23 10-26-08.png')
h,w,channel =src.shape
print(w)
print(h)
hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
hsv = cv2.inRange(hsv, (0, 0, 127), (180, 20, 225))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
iClose = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel)
iClose = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel)
cv2.imshow('hsv',hsv)
cv2.imshow('close',iClose)
tx = round(w/2)
ty = round(h/2)

while True:
    if(iClose[ty][tx]==0):
        tempy = ty +20
        break
    else:
        ty=ty+1
ty = round(h/2)
while True:
    if(iClose[tempy-30][tx]==0):
        tempx = tx -20
        break
    else:
        tx=tx-1
tx = round(w/2)
while True:
    if(iClose[ty][tx]==0):
        tempy1 = ty -20
        break
    else:
        ty=ty-1
while True:
    if(iClose[tempy1+30][tx]==0):
        tempx1 = tx +20
        break
    else:
        tx=tx+1
print(tempy)
print(tempx)
print(tempy1)
print(tempx1)
cv2.waitKey()
