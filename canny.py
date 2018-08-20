import cv2
import pyrealsense2 as rs
import numpy as np


def nothing(x):
    pass

# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
# pipeline.start(config)
cv2.namedWindow('image')
cv2.createTrackbar('L', 'image', 0, 255, nothing)
cv2.createTrackbar('H', 'image', 0, 255, nothing)

img = cv2.imread('img3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(img, 0, 150, apertureSize=3)

while 1:
    # frames = pipeline.wait_for_frames()
    # color_frame = frames.get_color_frame()
    # color_image = np.asanyarray(color_frame.get_data())
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', canny)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    l = cv2.getTrackbarPos('L', 'image')
    h = cv2.getTrackbarPos('H', 'image')
    # canny = cv2.GaussianBlur(img, (3, 3), 0)
    canny = cv2.Canny(img, l, h,apertureSize=3)
