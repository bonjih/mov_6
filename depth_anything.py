import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
from functools import cache


@cache
def load_depth_models():
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
    return image_processor, model


def process_frame(frame, inputs, model):


    # Perform depth estimation
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth_map = Image.fromarray(formatted)

    # Convert depth map to BGR for OpenCV
    depth_map_bgr = cv2.cvtColor(np.array(depth_map), cv2.COLOR_GRAY2BGR)

    # Resize depth map to match the frame size
    depth_map_resized = cv2.resize(depth_map_bgr, (frame.shape[1], frame.shape[0]))

    # Normalize depth map values to range [0, 255] and convert to uint8
    depth_map_normalized = (depth_map_resized * 255 / np.max(depth_map_resized)).astype(np.uint8)

    # Invert depth map to create mask
    mask = cv2.bitwise_not(cv2.cvtColor(depth_map_normalized, cv2.COLOR_BGR2GRAY))

    # Blend depth map with frame using bitwise_and
    blended_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return blended_frame


