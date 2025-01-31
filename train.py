import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged,
    RandRotate90d, RandFlipd, RandShiftIntensityd, RandAffine, ToTensord
)
from monai.data import Dataset, DataLoader
from topo import multi_class_topological_post_processing  # Import topology-based post-processing

# **Argument Parser**
def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net with Persistent Homology Post-Processing")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset (must contain 'train' and 'val' subdirectories)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--iterations", type=int, default=40000, help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for SGD")
    parser.add_argument("--momentum", type=float, default=0.99, help="Momentum for SGD optimizer")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--save-path", type=str, default="best_model.pth", help="Path to save the trained model")
    parser.add_argument("--apply-topo", action="store_true", help="Apply Persistent Homology post-processing during training")
    return parser.parse_args()

# **Load Data Paths**
def get_data_list(data_dir):
    """Loads image and label paths into a dictionary."""
    image_paths = sorted([os.path.join(data_dir, "images", f) for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith(".nii.gz")])
    label_paths = sorted([os.path.join(data_dir, "labels", f) for f in os.listdir(os.path.join(data_dir, "labels")) if f.endswith(".nii.gz")])
    return [{"image": img, "label": lbl} for img, lbl in zip(image_paths, label_paths)]

# **Training Function**
def train(args):
    # **Set Device**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # **Data Transforms**
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1, 2]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandAffine(keys=["image", "label"], prob=0.5, rotate_range=[0.1, 0.1, 0.1]),
        ToTensord(keys=["image", "label"]),
    ])

    # **Load Data**
    train_files = get_data_list('/home/bamdad/3D-PH-based-topological-loss-for-CNN-based-multi-class-segmentation-of-CMR-images/data/preprocessed_data')
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # **Define Model (3D U-Net)**
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,  # 5 segmentation classes (LA, RA, LV, RV, MY)
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # **Loss Function and Optimizer**
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # **Training Loop (Iteration-Based)**
    best_metric = -1
    iteration = 0
    while iteration < args.iterations:
        model.train()
        for batch in train_loader:
            if iteration >= args.iterations:
                break  # Stop if we've reached the desired number of iterations

            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # **Apply Topological Post-Processing One Image at a Time**
            if args.apply_topo:
                print("Applying Persistent Homology Post-Processing...")
                processed_outputs = []
                for i in range(inputs.shape[0]):  # Process each image separately
                    input_single = inputs[i].unsqueeze(0)  # Extract single image
                    output_single = outputs[i].unsqueeze(0)  # Extract single prediction
                    
                    # **Topological Post-Processing**
                    prior = {
                        (1,): (1, 0, 0),  # Class 1 should have 1 component, 0 loops, 0 voids
                        (2,): (1, 0, 0),  # Class 2 should have 1 component, 0 loops, 0 voids
                        (3,): (1, 0, 0),  # Class 3 should have 1 component, 0 loops, 0 voids
                        (4,): (1, 0, 0)   # Class 4 should have 1 component, 0 loops, 0 voids
                    }
                    refined_output = multi_class_topological_post_processing(
                        input_single, model, prior, lr=1e-5, mse_lambda=1.0
                    )
                    processed_outputs.append(refined_output)

                # **Stack Processed Outputs to Reconstruct the Batch**
                outputs = torch.cat(processed_outputs, dim=0)

            # **Compute Loss**
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 100 == 0:  # Print loss every 100 iterations
                print(f"Iteration {iteration}/{args.iterations}, Loss: {loss.item()}")

    print(f"Training complete. Best Dice Score: {best_metric}.")

# **Main Execution**
if __name__ == "__main__":
    args = parse_args()
    train(args)
