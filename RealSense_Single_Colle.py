import pyrealsense2 as rs
import numpy as np
import cv2


def RealSense():
    # Camera resolution
    fra = [640, 480]

    pipeline = rs.pipeline()
    config = rs.config()
    # Configure depth and color streams
    # Configure color camera
    config.enable_stream(rs.stream.color, fra[0], fra[1], rs.format.bgr8, 30)
    # Configure the infrared camera
    config.enable_stream(rs.stream.infrared, 1, fra[0], fra[1], rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, fra[0], fra[1], rs.format.y8, 30)
    # Configure depth image
    config.enable_stream(rs.stream.depth, fra[0], fra[1], rs.format.z16, 30)
    # Start streaming
    profile = pipeline.start(config)

    # Create an alignment object. rs.align enables us to align depth frames with other frames.
    # “align_to” specifies the stream type for which the depth frames are to be aligned.
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            # Align the depth box with the color box
            aligned_frames = align.process(frames)
            # Obtain the alignment frame
            aligned_depth_frame = aligned_frames.get_depth_frame()
            if not aligned_depth_frame:
                continue
            depth_frame = 50 * np.asanyarray(aligned_depth_frame.get_data())
            # Convert depth maps into pseudo-color images for easier viewing.
            depth_colormap = cv2.applyColorMap \
                (cv2.convertScaleAbs(depth_frame, alpha=0.008)
                 , cv2.COLORMAP_JET)
            cv2.imshow('1 depth', depth_colormap)
            cv2.imshow('2 depth', depth_frame)

            # color frames
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue
            color_frame = np.asanyarray(color_frame.get_data())
            cv2.imshow('2 color', color_frame)
            c = cv2.waitKey(1)

            # Pressing the ESC key closes the window (ESC's ASCII code is 27) and exits the loop.
            if c == 27:
                cv2.destroyAllWindows()
                break

    finally:
        # Stop streaming
        pipeline.stop()
