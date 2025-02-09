
import nibabel as nib
import os
import numpy as np

def find_label_files(data_dir):
    """Find all files in the data directory that end with 'label.nii.gz'."""
    # List all files in the directory
    files = os.listdir(data_dir)
    
    # Filter files that end with 'label.nii.gz'
    label_files = [f for f in files if f.endswith("label.nii.gz")]
    
    # Print the full paths of the matching files
    for label_file in label_files:
        print(os.path.join(data_dir, label_file))
    
    return label_files

def check_label_values(label_path):
    """Check the unique values in a label file."""
    label_data = nib.load(label_path).get_fdata()
    unique_values = np.unique(label_data)
    print(f"File: {label_path}")
    print(f"Unique values: {unique_values}")
    return unique_values

# Example usage
# Load the label file
label_path = "/home/bamdad/3D-PH-based-topological-loss-for-CNN-based-multi-class-segmentation-of-CMR-images/data/test/nii/ct_test_2001_label_encrypt_1mm.nii.gz"
label_data = nib.load(label_path).get_fdata()

# Print unique values and their counts
unique_values, counts = np.unique(label_data, return_counts=True)
print("Unique values:", unique_values)
print("Counts:", counts)