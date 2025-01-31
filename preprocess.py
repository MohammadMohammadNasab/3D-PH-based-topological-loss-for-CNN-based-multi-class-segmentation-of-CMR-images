import tqdm
import numpy as np
import SimpleITK as sitk
import os
import argparse

import SimpleITK as sitk
import numpy as np

def preprocess_image(image_path, label_path, target_spacing=(1.0, 1.0, 1.0), margin=10):
    """ Preprocess 3D medical images: resampling, normalization, cropping tightly around the foreground. """
    
    # Load the image and label
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # Resampling to target spacing
    def resample(image, target_spacing, is_label=False):
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        orig_size = np.array(image.GetSize(), dtype=np.int32)
        orig_spacing = np.array(image.GetSpacing())
        new_size = (orig_size * (orig_spacing / np.array(target_spacing))).astype(np.int32).tolist()
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        # Use nearest-neighbor interpolation for labels
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        return resampler.Execute(image)

    # Resample image and label
    image = resample(image, target_spacing, is_label=False)
    label = resample(label, target_spacing, is_label=True)

    # Convert images to NumPy arrays
    image_np = sitk.GetArrayFromImage(image)
    label_np = sitk.GetArrayFromImage(label)

    # **Find Foreground Bounding Box**
    # Use the original label values to find the bounding box
    coords = np.argwhere(label_np > 0)  # Find nonzero (foreground) pixels
    if len(coords) == 0:
        raise ValueError("No foreground found in the label volume.")
    
    min_coords = coords.min(axis=0)  # Min bounds
    max_coords = coords.max(axis=0)  # Max bounds

    # **Add a margin around the bounding box**
    min_coords = np.maximum(min_coords - margin, 0)
    max_coords = np.minimum(max_coords + margin, label_np.shape)

    # **Crop the image & label tightly around the foreground**
    cropped_image_np = image_np[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]
    cropped_label_np = label_np[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1], min_coords[2]:max_coords[2]]

    # Convert back to SimpleITK images
    cropped_image = sitk.GetImageFromArray(cropped_image_np)
    cropped_label = sitk.GetImageFromArray(cropped_label_np)

    # Set the new spacing, origin, and direction
    cropped_image.SetSpacing(target_spacing)
    cropped_label.SetSpacing(target_spacing)
    cropped_image.SetDirection(image.GetDirection())
    cropped_label.SetDirection(label.GetDirection())

    # Calculate new origin based on crop coordinates
    new_origin = [o + s * m for o, s, m in zip(image.GetOrigin(), image.GetSpacing(), min_coords)]
    cropped_image.SetOrigin(new_origin)
    cropped_label.SetOrigin(new_origin)

    return cropped_image, cropped_label

def preprocess_data_folder(data_dir, output_dir, target_spacing=(1.0, 1.0, 1.0), margin=10):
    """
    Preprocess all images and labels in the 'data' folder and save the results to the 'output_dir'.
    
    Args:
        data_dir (str): Path to the 'data' folder containing 'images' and 'labels' subfolders.
        output_dir (str): Path to save the preprocessed images and labels.
        target_spacing (tuple): Target spacing for resampling.
        margin (int): Margin to add around the foreground bounding box.
    """


    # Get image and label paths
    image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(os.path.join(data_dir)) if f.endswith("image.nii.gz")])
    label_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(os.path.join(data_dir)) if f.endswith("label.nii.gz")])

    # Process each image-label pair
    for image_path, label_path in tqdm.tqdm(zip(image_paths, label_paths), total=len(image_paths)):
        cropped_image, cropped_label = preprocess_image(image_path, label_path, target_spacing, margin)

        # Save preprocessed images and labels
        image_filename = os.path.basename(image_path)
        label_filename = os.path.basename(label_path)

        sitk.WriteImage(cropped_image, os.path.join(output_dir, image_filename))
        sitk.WriteImage(cropped_label, os.path.join(output_dir, label_filename))

def main():
    parser = argparse.ArgumentParser(description="Preprocess 3D medical images for segmentation.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the 'data' folder containing 'images' and 'labels' subfolders.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the preprocessed images and labels.")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Target spacing for resampling (e.g., 1.0 1.0 1.0).")
    parser.add_argument("--margin", type=int, default=10, help="Margin to add around the foreground bounding box.")
    args = parser.parse_args()

    # Call the preprocessing function with parsed arguments
    preprocess_data_folder(args.data_dir, args.output_dir, tuple(args.target_spacing), args.margin)

if __name__ == "__main__":
    main()
