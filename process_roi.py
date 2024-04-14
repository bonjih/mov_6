import os
from collections import deque
import cv2
import torch
import numpy as np


import global_params_variables
from depth_anything import load_depth_models, process_frame
from dust_detect import detect_blur_fft
from utils import get_centre, get_contours, create_rectangle_array
from vid_lables import draw_roi_poly, dusty_labels, timestamp, centre_labels

# setup
params = global_params_variables.ParamsDict()

motion_offset = params.get_value('motion_offset')


def extract_resize_roi(frame, roi_pts, target_size=(100, 100)):
    """
    Extract a region of interest (ROI) from the frame, resize it, and convert it to grayscale.

    Parameters:
        frame (numpy.ndarray): The original frame.
        roi_pts (list): List of points defining the region of interest (ROI).
        target_size (tuple): Target size of the ROI after resizing. Default is (100, 100).

    Returns:
        numpy.ndarray: The ROI resized and converted to grayscale.
        numpy.ndarray: The mask used to extract the ROI.
    """

    mask = np.zeros_like(frame[:, :, 0])

    cv2.fillPoly(mask, [roi_pts], (255, 255, 255))
    roi = cv2.bitwise_and(frame, frame, mask=mask)

    roi_resized = cv2.resize(roi, target_size)
    roi_image = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

    return roi_image, mask


def create_roi_mask(frame_shape, roi_points):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)
    return mask


class FrameProcessor:
    def __init__(self, roi_comp, output_dir):
        self.roi_comp = roi_comp
        self.prev_frames = {key: deque(maxlen=motion_offset) for key in roi_comp.rois}
        self.motion_frames = {key: 0 for key in roi_comp.rois}
        self.motion_start_frame = {key: 0 for key in roi_comp.rois}
        self.output_dir = output_dir

    def process_frame(self, frame, prev_frame, ts):
        mean, dusty = detect_blur_fft(frame)
        dusty_labels(frame, mean, dusty)
        timestamp(frame, ts)

        image_processor, model = load_depth_models()

        for roi_key in self.roi_comp.rois:
            roi = self.roi_comp.rois[roi_key]
            roi_points = roi.get_polygon_points()
            draw_roi_poly(frame, roi_key, roi_points)
            roi_mask = create_roi_mask(prev_frame.shape, roi_points)

            # Extract ROI from the frame using the mask
            roi_image, mask = extract_resize_roi(frame, roi_points, target_size=(100, 100))
            cnts = get_contours(roi_mask, mask)
            cX, cY = get_centre(cnts)

            # Prepare ROI image for the model
            roi_inputs = image_processor(images=roi_image, return_tensors="pt")

            # Process frame with depth map
            frame_with_depth = process_frame(roi_image, roi_inputs, model, ts)
            depth_resized = cv2.resize(frame_with_depth, (frame.shape[1], frame.shape[0]))

            # Resize the boolean mask to match the dimensions of the original frame
            resized_mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # Update frame with depth map for ROI
            frame[resized_mask != 0] = depth_resized[resized_mask != 0]
            centre_labels(frame, roi_key, cX, cY)
        return frame
