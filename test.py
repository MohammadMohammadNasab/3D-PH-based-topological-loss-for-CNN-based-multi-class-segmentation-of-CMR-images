import os
import argparse
import numpy as np
import torch
import SimpleITK as sitk
import cc3d
from monai.networks.nets import UNet
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd, ToTensord
from scipy.spatial.distance import directed_hausdorff
from betti_numbers import compute_betti_numbers
from metrics import dice_coefficient, hausdorff_distance, betti_error, topological_success_rate
from topo import multi_class_topological_post_processing  # Import topological post-processing

# **Apply Connected Component Analysis (CCA)**
def apply_cca(pred):
    """
    Apply Connected Component Analysis (CCA) to the predicted mask.
    Keeps only the largest connected component per class.
    
    Args:
        pred (numpy.ndarray): 3D predicted mask.

    Returns:
        numpy.ndarray: Processed mask with CCA applied.
    """
    processed_pred = np.zeros_like(pred)
    for class_id in np.unique(pred)[1:]:  # Skip background (0)
        mask = (pred == class_id).astype(np.uint8)
        labeled, num_features = cc3d.connected_components(mask, return_N=True)
        if num_features > 0:
            largest_component = np.argmax(np.bincount(labeled.flat)[1:]) + 1
            processed_pred[labeled == largest_component] = class_id
    return processed_pred

# **Test Function**
def evaluate_model(model_path, test_data_dir, output_report, apply_cca_flag, apply_topo_flag):
    """
    Evaluates the trained model on a test dataset with optional topological post-processing.

    Args:
        model_path (str): Path to the trained model checkpoint.
        test_data_dir (str): Directory containing test images and labels.
        output_report (str): Path to save the evaluation results.
        apply_cca_flag (bool): Whether to apply CCA to predictions.
        apply_topo_flag (bool): Whether to apply topological post-processing.
    """

    # **Load Model**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(spatial_dims=3, in_channels=1, out_channels=5, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # **Load Test Data**
    test_images = sorted([os.path.join(test_data_dir, "images", f) for f in os.listdir(os.path.join(test_data_dir, "images")) if f.endswith(".nii.gz")])
    test_labels = sorted([os.path.join(test_data_dir, "labels", f) for f in os.listdir(os.path.join(test_data_dir, "labels")) if f.endswith(".nii.gz")])

    test_files = [{"image": img, "label": lbl} for img, lbl in zip(test_images, test_labels)]

    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    # **Metrics Storage**
    all_dice_scores = []
    all_hausdorff = []
    all_betti_errors = []
    predictions_list = []
    ground_truths_list = []

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            label = batch["label"].cpu().numpy().astype(np.uint8)  # Convert to numpy for metric calculations

            # **Generate Prediction**
            output = model(image).cpu().numpy()
            print("Prediction shape:", output.shape)  # Debugging line
            pred = np.argmax(output, axis=1).squeeze(0)  # Convert to label map

            # **Apply CCA if requested**
            if apply_cca_flag:
                pred = apply_cca(pred)

            # **Apply Topological Post-Processing if requested**
            if apply_topo_flag:
                print("Applying Persistent Homological Loss-based Topological Post-Processing...")
                prior = {
                    (1,): (1, 0, 0),  # Class 1 should have 1 component, 0 loops, 0 voids
                    (2,): (1, 0, 0),  # Class 2 should have 1 component, 0 loops, 0 voids
                    (3,): (1, 0, 0),  # Class 3 should have 1 component, 0 loops, 0 voids
                    (4,): (1, 0, 0)   # Class 4 should have 1 component, 0 loops, 0 voids
                }
                pred_tensor = torch.tensor(pred, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                pred_tensor = multi_class_topological_post_processing(pred_tensor,
                                                                      model, prior, lr=1e-5, mse_lambda=1, num_its=100,
                                                                      construction='0', thresh=0.5, parallel=False)
                pred = torch.argmax(pred_tensor.cpu(), dim=1).squeeze(0).numpy()

            # **Compute Metrics**
            per_class_dice = [dice_coefficient((pred == i).astype(np.uint8), (label == i).astype(np.uint8)) for i in range(1, 5)]
            per_class_hausdorff = [hausdorff_distance((pred == i).astype(np.uint8), (label == i).astype(np.uint8)) for i in range(1, 5)]

            # **Generalized DSD (Mean of Per-Class Dice)**
            generalized_dice = np.mean(per_class_dice)

            # **Compute Betti Error**
            betti_err = betti_error(pred, label)

            # **Store Results**
            all_dice_scores.append(per_class_dice)
            all_hausdorff.append(per_class_hausdorff)
            all_betti_errors.append(betti_err)
            predictions_list.append(pred)
            ground_truths_list.append(label)

    # **Compute Topological Success Rate**
    topological_success = topological_success_rate(predictions_list, ground_truths_list)

    # **Write Results to File**
    with open(output_report, "w") as f:
        f.write("Evaluation Report\n")
        f.write("=================\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Dataset: {test_data_dir}\n")
        f.write(f"Apply CCA: {apply_cca_flag}\n")
        f.write(f"Apply Topological Post-Processing: {apply_topo_flag}\n\n")

        # Per-class metrics
        for i in range(4):  # Ignore background class 0
            f.write(f"Class {i+1} - Dice Score: {np.mean([d[i] for d in all_dice_scores]):.4f}\n")
            f.write(f"Class {i+1} - Hausdorff Distance: {np.mean([h[i] for h in all_hausdorff]):.4f}\n")

        # Generalized DSD
        f.write(f"\nGeneralized DSD (Mean Dice Score): {np.mean(generalized_dice):.4f}\n")

        # Betti Error
        f.write(f"\nAverage Betti Error: {np.mean(all_betti_errors):.4f}\n")

        # Topological Success Rate
        f.write(f"Topological Success Rate: {topological_success:.2f}%\n")

    print(f"Results saved to {output_report}")

# **Argument Parser**
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D Whole Heart Segmentation Model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--output-report", type=str, default="evaluation_results.txt", help="Path to save evaluation results")
    parser.add_argument("--apply-cca", action="store_true", help="Apply CCA to prediction or not")
    parser.add_argument("--apply-topo", action="store_true", help="Apply topological post-processing or not")

    args = parser.parse_args()
    evaluate_model(args.model_path, args.test_data, args.output_report, args.apply_cca, args.apply_topo)
