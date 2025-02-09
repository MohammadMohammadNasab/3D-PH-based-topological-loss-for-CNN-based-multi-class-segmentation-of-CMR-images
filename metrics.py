import gudhi as gd
from gudhi.cubical_complex import CubicalComplex
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def dice_coefficient(pred, gt, epsilon=1e-6):
    """Compute Dice Similarity Coefficient (DSC) for binary masks."""
    intersection =np.sum(pred * gt)
    return (2. * intersection + epsilon) / (np.sum(pred) + np.sum(gt) + epsilon)


def hausdorff_distance(pred, gt):
    """Compute Hausdorff Distance (HDD) between two binary masks."""
    pred_points = np.argwhere(pred > 0)
    gt_points = np.argwhere(gt > 0)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.inf  # No valid points, distance is undefined

    h1 = directed_hausdorff(pred_points, gt_points)[0]
    h2 = directed_hausdorff(gt_points, pred_points)[0]

    return max(h1, h2)


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


def compute_class_combinations_betti(segmentation, num_classes=5):
    """
    Compute Betti numbers for each foreground class (1-5) and their pairwise combinations.

    Args:
        segmentation (np.ndarray): 3D segmentation mask with multiple classes.
        num_classes (int): Number of foreground classes (excluding background 0).

    Returns:
        dict: Dictionary containing Betti numbers for each class and class pair.
    """
    combinations = {
        (c,): None for c in range(1, num_classes + 1)  # Single classes
    }

    # Compute single-class Betti numbers
    for class_idx in range(1, num_classes + 1):
        class_mask = (segmentation == class_idx).astype(np.uint8)
        if np.sum(class_mask) > 0:  # Compute only if the class exists
            combinations[(class_idx,)] = compute_betti_numbers_cubical(class_mask)
        else:
            combinations[(class_idx,)] = (0, 0, 0)

    # Compute pairwise combinations
    for i in range(1, num_classes):
        for j in range(i + 1, num_classes + 1):
            combined_mask = ((segmentation == i) | (segmentation == j)).astype(np.uint8)
            if np.sum(combined_mask) > 0:
                combinations[(i, j)] = compute_betti_numbers_cubical(combined_mask)
            else:
                combinations[(i, j)] = (0, 0, 0)

    return combinations


def betti_error_multi_class(pred, gt, num_classes=5):
    """
    Compute Betti Error (BE) for a multi-class 3D segmentation.

    Args:
        pred (numpy.ndarray): Predicted 3D segmentation mask.
        gt (numpy.ndarray): Ground truth 3D segmentation mask.
        num_classes (int): Number of foreground classes (default=5).

    Returns:
        float: Betti Error (sum of absolute differences across all classes and class combinations).
    """
    # Compute Betti numbers for each class and class pairs
    pred_betti = compute_class_combinations_betti(pred, num_classes)
    gt_betti = compute_class_combinations_betti(gt, num_classes)

    total_error = 0

    # Compute Betti error for single classes
    for c in range(1, num_classes + 1):
        total_error += sum(abs(np.array(pred_betti[(c,)]) - np.array(gt_betti[(c,)])))

    # Compute Betti error for class pairs
    for i in range(1, num_classes):
        for j in range(i + 1, num_classes + 1):
            total_error += sum(abs(np.array(pred_betti[(i, j)]) - np.array(gt_betti[(i, j)])))

    return total_error


def topological_success_rate(predictions, ground_truths, num_classes=5):
    """
    Compute the Topological Success (TS) rate for multi-class segmentation.

    Args:
        predictions (list of np.ndarray): List of predicted segmentation masks.
        ground_truths (listh of np.ndarray): List of ground truth segmentation masks.
        num_classes (int): Number of foreground classes.

    Returns:
        float: Percentage of cases where Betti Error is zero.
    """
    correct_count = sum(1 for pred, gt in zip(predictions, ground_truths) if betti_error_multi_class(pred, gt, num_classes) == 0)
    return (correct_count / len(predictions)) * 100 if len(predictions) > 0 else 0.0


def generalized_dice_score(predictions, ground_truths, num_classes=5, epsilon=1e-6):
    """
    Calculate the Generalized Dice Score for 3D multi-class segmentation.

    Args:
        predictions (np.ndarray): Predicted 3D segmentation volume (depth, height, width)
        ground_truths (np.ndarray): Ground truth 3D segmentation volume (depth, height, width)
        num_classes (int): Number of classes including background
        epsilon (float): Small constant to avoid division by zero

    Returns:
        float: Generalized Dice Score
    """
    dice_scores = []
    
    for c in range(num_classes):
        # Extract class-specific binary masks
        pred_c = (predictions == c).astype(float)
        gt_c = (ground_truths == c).astype(float)
        
        # Calculate intersection and sums
        intersection = np.sum(pred_c * gt_c)
        pred_sum = np.sum(pred_c)
        gt_sum = np.sum(gt_c)
        
        # Calculateh weight for this class (inverse of volume)
        w_c = 1 / (np.sum(gt_c) ** 2 + epsilon)
        
        # Calculate Dice score for this class
        class_dice = w_c * (2. * intersection + epsilon) / (pred_sum + gt_sum + epsilon)
        dice_scores.append(class_dice)
    
    #h Return mean Dice score across all classes
    return np.mean(dice_scores)


def batch_generalized_dice_score(predictions_batch, ground_truths_batch, num_classes=5):
    """
    Calculate average Generalized Dice Score across a batch of 3D volumes.

    Args:
        predictions_batch (list of np.ndarray): List of predicted 3Dh segmentation volumes
        ground_truths_batch (list of np.ndarray): List of ground truth 3D segmentation volumes
        num_classes (int): Number of classes including background

    Returns:
        float: Average Generalized Dice Score across theh batch
    """
    batch_scores = [
        generalized_dice_score(pred, gt, num_classes) 
        for pred, gt in zip(predictions_batch, ground_truths_batch)
    ]
    return np.mean(batch_scores)



