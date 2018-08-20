import cv2
import numpy as np

src = cv2.imread('img.jpg')
hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
print(hsv[345][300])
def nothing(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('LH','image',0,180,nothing)
cv2.createTrackbar('HH','image',0,180,nothing)
cv2.createTrackbar('LS','image',0,255,nothing)
cv2.createTrackbar('HS','image',0,255,nothing)
cv2.createTrackbar('LV','image',0,255,nothing)
cv2.createTrackbar('HV','image',0,255,nothing)

while(1):
    #src = cv2.GaussianBlur(src,(3,3),0)
    cv2.imshow('image',src)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    lh = cv2.getTrackbarPos('LH', 'image')
    hh = cv2.getTrackbarPos('HH', 'image')
    ls = cv2.getTrackbarPos('LS', 'image')
    hs = cv2.getTrackbarPos('HS', 'image')
    lv = cv2.getTrackbarPos('LV', 'image')
    hv = cv2.getTrackbarPos('HV', 'image')

    src = cv2.inRange(hsv, (lh, ls, lv), (hh, hs, hv))
    # hv=115


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# image=cv2.imread('/media/yingkai/Ubuntu-GNOM/ubuntu_gnom_back/Pictures/Screenshot from 2018-07-26 22-33-26.png')
# HSV=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
# def getpos(event,x,y,flags,param):
#     if event==cv2.EVENT_LBUTTONDOWN:
#         print(HSV[y,x])
# #th2=cv2.adaptiveThreshold(imagegray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# cv2.imshow("imageHSV",HSV)
# cv2.imshow('image',image)
# cv2.setMouseCallback("imageHSV",getpos)
# cv2.waitKey(0)