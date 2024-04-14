import cv2
from matplotlib import pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from functools import cache
from collections import deque

# Check if GPU is available
from stats import plot_histogram, calculate_mean_and_std, depth_perception_metrics, calculate_contrast, \
    time_series_analysis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deque_size = 10
depth_sequence = deque(maxlen=deque_size)

@cache
def load_depth_models():
    image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
    return image_processor, model


def process_frame(frame, inputs, model, ts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs_gpu = {key: tensor.to(device) for key, tensor in inputs.items()}

    # Perform depth estimation
    with torch.no_grad():
        outputs = model(**inputs_gpu)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    )

    depth_sequence.append(prediction)

    if len(depth_sequence) == deque_size:

        depth_means, depth_variances = time_series_analysis(depth_sequence)
        depth_means = depth_means[1:]
        depth_variances = depth_variances[1:]

        # Calculate time steps in seconds
        time_steps = np.arange(len(depth_sequence)) * (ts / 1000)
        time_steps = time_steps[1:]

        # plt.figure(figsize=(10, 5))
        # plt.subplot(2, 1, 1)
        # plt.plot(time_steps, depth_means, marker='o')
        # plt.xlabel('Time (seconds)')
        # plt.ylabel('Mean Depth')
        # plt.title('Mean Depth over Time')
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(time_steps, depth_variances, marker='o')
        # plt.xlabel('Time (seconds)')
        # plt.ylabel('Depth Variance')
        # plt.title('Depth Variance over Time')

        plt.tight_layout()
        plt.show()



    output = prediction.squeeze().cpu().numpy()

    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth_map = cv2.cvtColor(formatted, cv2.COLOR_GRAY2BGR)

    # Resize depth map to match the frame size
    depth_map_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # Normalize depth map values to range [0, 255] and convert to uint8
    depth_map_normalized = (depth_map_resized * 255 / np.max(depth_map_resized)).astype(np.uint8)

    # Invert depth map to create mask
    mask = cv2.bitwise_not(cv2.cvtColor(depth_map_normalized, cv2.COLOR_BGR2GRAY))

    # Blend depth map with frame using bitwise_and
    blended_frame = cv2.bitwise_and(frame, frame, mask=mask)

    return blended_frame


