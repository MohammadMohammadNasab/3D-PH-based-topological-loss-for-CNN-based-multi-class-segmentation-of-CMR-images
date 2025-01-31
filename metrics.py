import gudhi as gd
from gudhi.cubical_complex import CubicalComplex
import cc3d
from scipy.spatial.distance import directed_hausdorff
import numpy as np


def dice_coefficient(pred, gt, epsilon=1e-6):
    """
    Compute Dice Similarity Coefficient (DSC) for binary masks.

    Args:
        pred (numpy.ndarray): Predicted binary segmentation mask.
        gt (numpy.ndarray): Ground truth binary segmentation mask.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: Dice Similarity Coefficient (DSC)
    """
    intersection = np.sum(pred * gt)
    return (2. * intersection + epsilon) / (np.sum(pred) + np.sum(gt) + epsilon) 


def hausdorff_distance(pred, gt):
    """
    Compute Hausdorff Distance (HDD) between two binary masks.

    Args:
        pred (numpy.ndarray): Predicted binary segmentation mask.
        gt (numpy.ndarray): Ground truth binary segmentation mask.

    Returns:
        float: Hausdorff Distance
    """
    pred_points = np.argwhere(pred > 0)
    gt_points = np.argwhere(gt > 0)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.inf  # No valid points, distance is undefined

    h1 = directed_hausdorff(pred_points, gt_points)[0]
    h2 = directed_hausdorff(gt_points, pred_points)[0]

    return max(h1, h2)



def betti_error(pred, gt):
    """
    Compute Betti Error (BE) by comparing 3D topological structures.

    Args:
        pred (numpy.ndarray): Predicted binary segmentation mask.
        gt (numpy.ndarray): Ground truth binary segmentation mask.

    Returns:
        int: Betti Error (BE)
    """
    # Compute Betti numbers for prediction and ground truth
    b0_pred, b1_pred, b2_pred = compute_betti_numbers(pred)
    b0_gt, b1_gt, b2_gt = compute_betti_numbers(gt)

    # Betti Error is the sum of absolute differences across Betti numbers
    return abs(b0_pred - b0_gt) + abs(b1_pred - b1_gt) + abs(b2_pred - b2_gt)

# Example Usage:
# betti_err = betti_error(pred_mask, gt_mask)


def topological_success_rate(predictions, ground_truths):
    """
    Compute the Topological Success (TS) rate.

    Args:
        predictions (list of numpy.ndarray): List of predicted segmentation masks.
        ground_truths (list of numpy.ndarray): List of ground truth segmentation masks.

    Returns:
        float: Percentage of cases where Betti Error is zero.
    """
    correct_count = sum(1 for pred, gt in zip(predictions, ground_truths) if betti_error(pred, gt) == 0)
    return (correct_count / len(predictions)) * 100 if len(predictions) > 0 else 0.0

import numpy as np

def compute_betti_numbers_cubical(mask):
    """
    Compute 3D Betti numbers (b0, b1, b2) using a Cubical Complex approach.

    Args:
        mask (numpy.ndarray): 3D binary segmentation mask.

    Returns:
        tuple: (b0, b1, b2) - Betti numbers representing connected components, loops, and voids.
    """
    # Convert binary mask into a 3D function (flip values to treat 1s as foreground)
    cubical_data = 1 - mask.astype(np.float32)

    # Build a Cubical Complex
    cubical_complex = CubicalComplex(top_dimensional_cells=cubical_data)

    # Compute Persistent Homology
    cubical_complex.persistence()

    # Get Betti numbers
    b0 = cubical_complex.betti_number(0)  # Number of connected components
    b1 = cubical_complex.betti_number(1)  # Number of loops (1D holes)
    b2 = cubical_complex.betti_number(2)  # Number of voids (2D holes)

    return b0, b1, b2
