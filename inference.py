import os
import argparse
import numpy as np
import torch
import SimpleITK as sitk
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirst, NormalizeIntensity, ToTensor, Resize
from topo import multi_class_topological_post_processing
import re

def apply_cca(pred):
    """Apply Connected Component Analysis (CCA) to the predicted mask."""
    processed_pred = np.zeros_like(pred)
    for class_id in np.unique(pred)[1:]:  # Skip background (0)
        mask = (pred == class_id).astype(np.uint8)
        labeled, num_features = cc3d.connected_components(mask, return_N=True)
        if num_features > 0:
            largest_component = np.argmax(np.bincount(labeled.flat)[1:]) + 1
            processed_pred[labeled == largest_component] = class_id
    return processed_pred

def format_output_filename(input_path):
    """Format output filename according to required convention."""
    basename = os.path.basename(input_path)
    
    # First try the test pattern
    test_match = re.search(r'mr_test_20(\d{2})', basename)
    if test_match:
        case_number = test_match.group(1)
        return f"mr_test_20{case_number}_label.nii.gz"
    
    # Try training pattern and convert to test pattern
    train_match = re.search(r'mr_train_(\d{4})', basename)
    if train_match:
        case_number = str(int(train_match.group(1)) - 1000).zfill(2)  # Convert 1001 to 01
        return f"mr_test_20{case_number}_label.nii.gz"
        
    # If neither pattern matches, use a default pattern with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m")  # Get current year and month
    return f"mr_test_20{timestamp}_label.nii.gz"

def resample_image(image_sitk, target_spacing=(1.0, 1.0, 1.0)):
    """Use SimpleITK to resample the image to target spacing."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    orig_size = np.array(image_sitk.GetSize(), dtype=np.int32)
    orig_spacing = np.array(image_sitk.GetSpacing(), dtype=np.float32)
    new_size = (orig_size * (orig_spacing / np.array(target_spacing))).astype(np.int32).tolist()
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetOutputDirection(image_sitk.GetDirection())
    # Use linear interpolation for single-channel intensity
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image_sitk)

def resample_label_to_reference(label_sitk, reference_sitk):
    """
    Resample label_sitk (in 1x1x1 space) to match reference_sitk geometry.
    Uses nearest-neighbor to avoid interpolating segmentation classes.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(label_sitk)

def resample_to_reference(moving_image, reference_image, is_label=False):
    """
    Resample moving_image to match reference_image's geometry precisely.
    
    Args:
        moving_image: SimpleITK image to resample
        reference_image: SimpleITK image with desired geometry
        is_label: If True, use nearest neighbor interpolation
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(moving_image)

def resize_to_fixed_size(image_sitk, target_size=(192, 160, 160), is_label=False):
    """
    Resize SimpleITK image to fixed dimensions using MONAI's Resize transform
    """
    # Convert to numpy array
    image_np = sitk.GetArrayFromImage(image_sitk)
    
    # Add channel dimension
    image_np = np.expand_dims(image_np, axis=0)
    
    # Create MONAI Resize transform
    resize = Resize(
        spatial_size=target_size,
        mode='nearest' if is_label else 'trilinear'
    )
    
    # Apply resize
    resized_np = resize(image_np)
    
    # Remove channel dimension and convert back to SimpleITK
    resized_sitk = sitk.GetImageFromArray(resized_np[0])
    
    # Instead of CopyInformation, set the metadata manually with adjusted spacing
    orig_spacing = np.array(image_sitk.GetSpacing())
    orig_size = np.array(image_sitk.GetSize())
    new_size = np.array(resized_sitk.GetSize())
    
    # Calculate new spacing to maintain physical size
    new_spacing = (orig_spacing * orig_size) / new_size
    
    resized_sitk.SetSpacing(new_spacing.tolist())
    resized_sitk.SetDirection(image_sitk.GetDirection())
    resized_sitk.SetOrigin(image_sitk.GetOrigin())
    
    return resized_sitk

def run_inference(model_path, input_path, output_dir, original_image_path=None,
                  apply_cca_flag=False, apply_topo_flag=False):
    """Run inference on a 3D medical image."""
    os.makedirs(output_dir, exist_ok=True)

    # Load original image (required for correct resampling)
    original_image = sitk.ReadImage(input_path)
    original_size = original_image.GetSize()[::-1]  # Reverse for numpy convention
    
    # Create 1mm reference image with same geometry
    reference_size = original_image.GetSize()
    reference_spacing = (1.0, 1.0, 1.0)
    reference_direction = original_image.GetDirection()
    reference_origin = original_image.GetOrigin()
    
    reference_image = sitk.Image(reference_size, original_image.GetPixelID())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)
    
    # Resample input to 1mm spacing for model
    input_image_1mm = resample_to_reference(original_image, reference_image)
    
    # Then resize to fixed dimensions (192, 160, 160)
    input_image_resized = resize_to_fixed_size(input_image_1mm, target_size=(192, 160, 160))

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=8,  # 
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()

    # Convert to numpy and apply transforms
    image = sitk.GetArrayFromImage(input_image_resized)
    # Add channel dimension manually before applying other transforms
    image = np.expand_dims(image, axis=0)  # Add channel dimension
    
    # Modify transforms to skip EnsureChannelFirst since we're doing it manually
    transforms = Compose([
        NormalizeIntensity(nonzero=True, channel_wise=True),
        ToTensor()
    ])

    image = transforms(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Generate prediction
    print("Running inference...")
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Apply post-processing if requested
    if apply_cca_flag:
        print("Applying CCA...")
        pred = apply_cca(pred)

    if apply_topo_flag:
        print("Applying topological post-processing...")
        prior = {
            (1,): (1, 0, 0),
            (2,): (1, 0, 0),
            (3,): (1, 0, 0),
            (4,): (1, 0, 0)
        }
        pred_tensor = torch.tensor(pred, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        pred_tensor = multi_class_topological_post_processing(
            pred_tensor, model, prior,
            lr=1e-5, mse_lambda=1, num_its=100,
            construction='0', thresh=0.5, parallel=False
        )
        pred = torch.argmax(pred_tensor.cpu(), dim=1).squeeze(0).numpy()

    # Convert prediction to SITK image with 1mm properties
    pred_sitk = sitk.GetImageFromArray(pred.astype(np.uint16))
    pred_sitk.CopyInformation(input_image_resized)
    
    # First resize back to 1mm size
    pred_1mm = resize_to_fixed_size(pred_sitk, 
                                   target_size=input_image_1mm.GetSize()[::-1],
                                   is_label=True)
    
    # Then resample prediction back to original space
    final_pred = resize_to_fixed_size(pred_sitk, 
                                    target_size=original_size,  # Use original size
                                    is_label=True)
    
    # Copy the original image's metadata
    final_pred.SetSpacing(original_image.GetSpacing())
    final_pred.SetDirection(original_image.GetDirection())
    final_pred.SetOrigin(original_image.GetOrigin())
    
    # Save with original image properties
    output_filename = format_output_filename(input_path)
    output_path = os.path.join(output_dir, output_filename)
    print(f"Saving prediction to {output_path}")
    print(f"Original image size: {original_image.GetSize()}")
    print(f"Final prediction size: {final_pred.GetSize()}")
    sitk.WriteImage(final_pred, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Medical Image Segmentation Inference")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input-path", type=str, required=True, help="Path to input 3D image")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save prediction")
    parser.add_argument("--original-image", type=str, help="Optional path to original image for setting output metadata")
    parser.add_argument("--apply-cca", action="store_true", help="Apply CCA to prediction")
    parser.add_argument("--apply-topo", action="store_true", help="Apply topological post-processing")

    args = parser.parse_args()
    run_inference(
        args.model_path,
        args.input_path,
        args.output_dir,
        args.original_image,
        args.apply_cca,
        args.apply_topo
    )
