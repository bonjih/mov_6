import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def calculate_mean_and_std(tensor):
    """
    Calculate the mean and standard deviation of a PyTorch tensor.

    Args:
    - tensor: Input PyTorch tensor

    Returns:
    - mean: Mean value of the tensor
    - std: Standard deviation of the tensor
    """
    mean = tensor.mean().item()
    std = tensor.std().item()
    return mean, std


# # Example usage:
# # Assuming `prediction` is your PyTorch tensor
# mean, std = calculate_mean_and_std(prediction)
# print("Mean:", mean)
# print("Standard Deviation:", std)
#


def plot_histogram(tensor, num_bins=50):
    """
    Plot a histogram of the values in a PyTorch tensor.

    Args:
    - tensor: Input PyTorch tensor
    - num_bins: Number of bins for the histogram (default is 50)
    """
    # Flatten the tensor to 1D
    flattened_tensor = tensor.flatten()

    # Convert the tensor to a NumPy array
    numpy_array = flattened_tensor.numpy()

    # Plot the histogram
    plt.hist(numpy_array, bins=num_bins)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Tensor Values')
    plt.show()


def depth_perception_metrics(depth_map_tensor):
    """
    Compute depth perception metrics for a depth map represented as a PyTorch tensor.

    Args:
    - depth_map_tensor: Input PyTorch tensor representing the depth map

    Returns:
    - depth_range: Range of depth values in the depth map
    - depth_transitions: Number of depth transitions (changes in depth values)
    - occlusion_cues: Number of occlusion cues (e.g., sharp depth discontinuities)
    """
    # Flatten the tensor to 1D
    flattened_tensor = depth_map_tensor.flatten()

    # Compute depth range
    depth_range = torch.max(flattened_tensor) - torch.min(flattened_tensor)

    # Compute depth transitions
    depth_transitions = torch.sum(flattened_tensor[:-1] != flattened_tensor[1:]).item()

    # Compute occlusion cues (assuming occlusion cues occur where depth transitions are abrupt)
    # This can be further refined based on specific requirements and characteristics of the depth map
    occlusion_cues = torch.sum(torch.abs(flattened_tensor[:-1] - flattened_tensor[1:]) > 1).item()

    return depth_range.item(), depth_transitions, occlusion_cues


def calculate_contrast(tensor):
    """
    Calculate the contrast of an image represented as a PyTorch tensor using variance.

    Args:
    - tensor: Input PyTorch tensor representing the image

    Returns:
    - contrast: Contrast of the image
    """

    # range
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    pixel_range = max_val - min_val

    # Flatten the tensor to 1D
    flattened_tensor = tensor.flatten()

    # Compute the variance of the pixel values
    variance = torch.var(flattened_tensor)
    std_dev = torch.std(flattened_tensor)

    # stats
    var = variance.item()
    std = std_dev.item()
    pix_range = pixel_range.item()
    return var, std, pix_range


def edge_detection_canny(tensor, low_threshold=50, high_threshold=150):
    """
    Perform edge detection using the Canny edge detection algorithm on an image represented as a PyTorch tensor.

    Args:
    - tensor: Input PyTorch tensor representing the image
    - low_threshold: Lower threshold for edge detection (default is 50)
    - high_threshold: Upper threshold for edge detection (default is 150)

    Returns:
    - edge_image: PyTorch tensor representing the edge-detected image
    """
    # Convert PyTorch tensor to NumPy array and then to OpenCV's expected format (uint8)
    image_np = tensor.cpu().numpy().squeeze().astype(np.uint8)

    # Apply Canny edge detection algorithm
    edge_image_np = cv2.Canny(image_np, low_threshold, high_threshold)

    # Convert the resulting NumPy array back to a PyTorch tensor
    edge_image_tensor = torch.tensor(edge_image_np, dtype=torch.float32)

    return edge_image_tensor


def time_series_analysis(depth_sequence):
    """
    Perform time-series analysis on a sequence of depth images represented as PyTorch tensors.

    Args:
    - depth_sequence: List of PyTorch tensors representing the depth images over time

    Returns:
    - depth_means: List of mean depth values for each time step
    - depth_variances: List of variances of depth values for each time step
    """
    depth_means = []
    depth_variances = []

    for depth_image_tensor in depth_sequence:
        # Flatten the tensor to 1D
        flattened_tensor = depth_image_tensor.flatten()

        # Compute mean and variance of depth values
        depth_mean = torch.mean(flattened_tensor).item()
        depth_variance = torch.var(flattened_tensor).item()

        depth_means.append(depth_mean)
        depth_variances.append(depth_variance)

    return depth_means, depth_variances
