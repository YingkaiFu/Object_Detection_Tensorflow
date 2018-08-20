import pyrealsense2 as rs
import numpy as np
import cv2
import time

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

profile = pipeline.start(config)
dev = profile.get_device()
sensor = dev.query_sensors()
t = 1
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

clipping_distance_in_meters = 1.2  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)

# 相机刚开始工作的时候色温不对,因此需要进行预热
while True:

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    # aligned_depth_frame is a 640x480 depth image

    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    depth_image_3d = np.dstack(
        (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
    # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

    # Render images
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # if t == 100:
    #     cv2.imwrite('img.jpg', color_image)
    #     images_stack = np.hstack((color_image, depth_colormap))
    #     break
    # t = t + 1
    images_stack = np.hstack((color_image, depth_colormap))
    cv2.imshow('hello',images_stack)
    cv2.waitKey(1)
