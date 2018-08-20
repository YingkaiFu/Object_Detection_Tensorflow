import pyrealsense2 as rs
import cv2
import numpy as np
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
#        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
 #       images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey()

finally:

    # Stop streaming
    pipeline.stop()