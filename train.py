import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from monai.transforms import (
    RandAxisFlipd, Resized, Lambdad,
    Compose, LoadImaged,
    RandRotate90d, RandShiftIntensityd, ToTensord
)
from monai.data import Dataset, DataLoader
from topo import multi_class_topological_post_processing  # Import topology-based post-processing

label_mapping = {
    0: 0,   # Background
    205: 1, # Myocardium (MY)
    420: 2, # Left Atrium (LA)
    421: 2, # Mislabelled data considerd as  Left Atrium (LA)
    500: 3, # Left Ventricle (LV)
    550: 4, # Right Atrium (RA)
    600: 5, # Right Ventricle (RV)
    820: 6, # Ascending Aorta (AA)
    850: 7  # Pulmonary Artery (PA)
}

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
    image_paths = sorted([os.path.join(data_dir,f) for f in os.listdir(os.path.join(data_dir)) if f.endswith("_image.nii.gz")])
    label_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(os.path.join(data_dir)) if f.endswith("_label.nii.gz")])
    return [{"image": img, "label": lbl} for img, lbl in zip(image_paths, label_paths)]

def map_labels(label, mapping):
    """Maps MM-WHS intensity values to class indices."""
    for raw_value, class_index in mapping.items():
        label[label == raw_value] = class_index
    return label

# **Training Function**
def train(args):
    # **Set Device**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # **Data Transforms**
    train_transforms = Compose([
    LoadImaged(keys=["image", "label"], ensure_channel_first=True),
    Lambdad(keys="label", func=lambda x: map_labels(x, label_mapping)),
    RandAxisFlipd(keys=["image", "label"], prob=0.5),  # Flip along one random axis
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(1, 2)),  # Rotate in axial plane
    RandShiftIntensityd(keys="image", offsets=0.15, prob=0.5),
    Resized(
        keys=["image", "label"],
        spatial_size=(192, 160, 160),
        mode=("trilinear", "nearest")  # Different modes for image/label
    ),
    ToTensord(keys=["image", "label"]),
    Lambdad(keys="label", func=lambda x: x.squeeze(1).long()),
])

    # **Load Data**
    train_files = get_data_list(args.data_dir)
    train_ds = Dataset(train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # **Define Model (3D U-Net)**
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=8,  # 
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # **Loss Function and Optimizer**
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # **Training Loop (Iteration-Based)**
    best_metric = float('inf')
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
            loss = loss_function(outputs, labels.squeeze(1).long())
            loss.backward()
            optimizer.step()

            iteration += 1
            if iteration % 100 == 0 and loss.item() < best_metric:
                best_metric = loss.item()  # Print loss every 100 iterations
                print(f"Iteration {iteration}/{args.iterations}, Loss: {loss.item()}")
                # Save model checkpoint
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'best_metric': best_metric,
                }, args.save_path)
                

    print(f"Training complete. Best Dice Score: {best_metric}.")

# **Main Execution**
if __name__ == "__main__":
    args = parse_args()
    train(args)
