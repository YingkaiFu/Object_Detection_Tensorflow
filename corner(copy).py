import cv2
import numpy as np


def getCorner(img):
    src = cv2.imread(img)
    hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
    out = cv2.inRange(hsv,(0,0,129),(360,22,255))
    fh,fw = out.shape
    image, cnts, hierarchy = cv2.findContours(out,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    cnt = sorted(cnts,key=cv2.contourArea)
    print(len(cnt[0]))
    count=0
    for i in range(fh):
            for j in range(fw):
                if out[i][j]==255:
                    count = count+1
    print(count)


getCorner('test.jpg')