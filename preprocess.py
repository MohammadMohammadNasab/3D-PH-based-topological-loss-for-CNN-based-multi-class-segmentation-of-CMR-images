import cc3d
import numpy as np
import SimpleITK as sitk

def preprocess_image(image_path, label_path, target_spacing=(1.0, 1.0, 1.0), margin=10):
    """ Preprocess 3D medical images: resampling, normalization, cropping tightly around the foreground. """
    
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # Resampling to target spacing
    def resample(image, target_spacing):
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        orig_size = np.array(image.GetSize(), dtype=np.int32)
        orig_spacing = np.array(image.GetSpacing())
        new_size = (orig_size * (orig_spacing / np.array(target_spacing))).astype(np.int32).tolist()
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear if image.GetPixelID() != sitk.sitkUInt8 else sitk.sitkNearestNeighbor)
        return resampler.Execute(image)

    image = resample(image, target_spacing)
    label = resample(label, target_spacing)

    # Convert images to NumPy arrays
    image_np = sitk.GetArrayFromImage(image)
    label_np = sitk.GetArrayFromImage(label)

    # **Find Foreground Bounding Box**
    labels_out = cc3d.connected_components(label_np)  # Label connected components
    coords = np.argwhere(labels_out > 0)  # Find nonzero (foreground) pixels
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

    # **Preserve metadata** (spacing, origin, direction)
    cropped_image.CopyInformation(image)
    cropped_label.CopyInformation(label)

    return cropped_image, cropped_label
