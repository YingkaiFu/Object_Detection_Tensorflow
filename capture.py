import pyrealsense2 as rs
import numpy as np
import cv2
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
pipeline.start(config)

# t = 1
# while True:
#     frames = pipeline.wait_for_frames()
#     color_frame = frames.get_color_frame()
#
#     # Convert images to numpy arrays
#     color_image = np.asanyarray(color_frame.get_data())
#     if t == 100:
#         cv2.imshow()
#         cv2.imwrite('img.jpg', color_image)capture.py:17
#         break
#     t = t + 1
i=401
while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow('img', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        name = '/home/yingkai/Data/capture/test'+str(i)+'.jpg'
        cv2.imwrite(name, color_image)
        i=i+1


