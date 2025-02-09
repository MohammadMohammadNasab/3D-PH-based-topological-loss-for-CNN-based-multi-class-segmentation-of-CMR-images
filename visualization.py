import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
import os

def resample_label_to_reference(label_sitk, reference_sitk):
    """Resample segmentation label to match the reference image's shape and metadata."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_sitk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(label_sitk)

class SegmentationViewer:
    def __init__(self, image_path, pred_segmentation_path, true_label_path=None, output_dir=None):
        """Initialize viewer with original image, predicted segmentation, and optionally true label"""
        # Set output directory
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Load all images
        ref_img_sitk = sitk.ReadImage(image_path)
        pred_seg_sitk = sitk.ReadImage(pred_segmentation_path)
        
        # Convert to numpy arrays
        self.image = sitk.GetArrayFromImage(ref_img_sitk)
        self.pred_segmentation = sitk.GetArrayFromImage(pred_seg_sitk)
        
        # Initialize slice number before creating the slider
        self.slice_num = self.image.shape[0] // 2  # Start from middle slice
        
        # Setup the figure
        if true_label_path:
            true_seg_sitk = sitk.ReadImage(true_label_path)
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
            self.has_true_label = True
            self.true_segmentation = sitk.GetArrayFromImage(true_seg_sitk)
        else:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
            self.has_true_label = False
            
        # Resample segmentations if needed
        if ref_img_sitk.GetSize() != pred_seg_sitk.GetSize():
            print("Resampling predicted segmentation to match image...")
            pred_seg_sitk = resample_label_to_reference(pred_seg_sitk, ref_img_sitk)
            
        if self.has_true_label and ref_img_sitk.GetSize() != true_seg_sitk.GetSize():
            print("Resampling true label to match image...")
            true_seg_sitk = resample_label_to_reference(true_seg_sitk, ref_img_sitk)
        
        # Verify shapes
        if self.image.shape != self.pred_segmentation.shape:
            raise ValueError("Image and predicted segmentation shapes differ after resampling.")
        if self.has_true_label and self.image.shape != self.true_segmentation.shape:
            raise ValueError("Image and true label shapes differ after resampling.")
        
        # Normalize image for better visualization
        self.image = (self.image - self.image.min()) / (self.image.max() - self.image.min())
        
        # Create colormaps
        self.seg_colors = {
            0: [0, 0, 0, 0],      # Background (transparent)
            1: [1, 0, 0, 0.5],    # Myocardium (red)
            2: [0, 1, 0, 0.5],    # Left Atrium (green)
            3: [0, 0, 1, 0.5],    # Left Ventricle (blue)
            4: [1, 1, 0, 0.5],    # Right Atrium (yellow)
            5: [1, 0, 1, 0.5],    # Right Ventricle (magenta)
            6: [0, 1, 1, 0.5],    # Ascending Aorta (cyan)
            7: [1, 0.5, 0, 0.5]   # Pulmonary Artery (orange)
        }
        
        # Setup the slider last
        self.slider_ax = plt.axes([0.15, 0.02, 0.7, 0.03])
        self.slider = Slider(
            self.slider_ax, 'Slice', 0, self.image.shape[0]-1,
            valinit=self.slice_num, valstep=1
        )
        self.slider.on_changed(self.update_slice)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.5) 
                         for color in list(self.seg_colors.values())[1:]]
        legend_labels = ['Myocardium', 'Left Atrium', 'Left Ventricle', 
                        'Right Atrium', 'Right Ventricle', 'Ascending Aorta', 
                        'Pulmonary Artery']
        
        # Move legend to the last subplot that exists
        if self.has_true_label:
            self.ax3.legend(legend_elements, legend_labels, 
                          loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            self.ax2.legend(legend_elements, legend_labels, 
                          loc='center left', bbox_to_anchor=(1, 0.5))

        # Update window title to show what we're viewing
        title = 'Segmentation Comparison' if self.has_true_label else 'Segmentation Viewer'
        self.fig.suptitle(title, fontsize=16)
        
        # Initial update
        self.update_slice(self.slice_num)

    def save_current_slice(self):
        """Save the current slice visualization"""
        if not self.output_dir:
            print("No output directory specified")
            return
            
        filename = f"slice_{self.slice_num:03d}.png"
        filepath = os.path.join(self.output_dir, filename)
        self.fig.savefig(filepath, bbox_inches='tight', dpi=300)
        print(f"Saved slice to {filepath}")
        
    def save_all_slices(self):
        """Save all slices as separate images"""
        if not self.output_dir:
            print("No output directory specified")
            return
            
        for slice_idx in range(self.image.shape[0]):
            self.slider.set_val(slice_idx)  # This will trigger update_slice
            self.save_current_slice()

    def update_slice(self, val):
        self.slice_num = int(val)
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        if self.has_true_label:
            self.ax3.clear()
        
        # Plot original image in all subplots as background
        self.ax1.imshow(self.image[self.slice_num], cmap='gray')
        self.ax2.imshow(self.image[self.slice_num], cmap='gray')
        if self.has_true_label:
            self.ax3.imshow(self.image[self.slice_num], cmap='gray')
        
        # Plot segmentation overlays
        def overlay_segmentation(ax, seg_data, title):
            for label, color in self.seg_colors.items():
                mask = seg_data == label
                if mask.any():
                    ax.imshow(np.ma.masked_where(~mask, mask),
                            cmap=plt.matplotlib.colors.ListedColormap([color]),
                            alpha=0.5)
            ax.set_title(title)
            ax.axis('off')
        
        # Apply overlays
        self.ax1.set_title(f'Original Image (Slice {self.slice_num})')
        self.ax1.axis('off')
        
        overlay_segmentation(self.ax2, self.pred_segmentation[self.slice_num],
                           'Predicted Segmentation')
        
        if self.has_true_label:
            overlay_segmentation(self.ax3, self.true_segmentation[self.slice_num],
                               'Ground Truth')
        
        # Ensure tight layout
        self.fig.tight_layout()
        plt.draw()
        
        if self.output_dir:
            self.save_current_slice()

def main():
    parser = argparse.ArgumentParser(description='Visualize 3D heart segmentation')
    parser.add_argument('--image', type=str, required=True, help='Path to original image')
    parser.add_argument('--prediction', type=str, required=True, help='Path to predicted segmentation')
    parser.add_argument('--true-label', type=str, help='Path to true label (optional)')
    parser.add_argument('--output-dir', type=str, help='Directory to save visualization plots')
    parser.add_argument('--save-all', action='store_true', help='Save all slices')
    args = parser.parse_args()
    
    viewer = SegmentationViewer(args.image, args.prediction, args.true_label, args.output_dir)
    
    if args.save_all:
        viewer.save_all_slices()
    elif args.output_dir:
        viewer.save_current_slice()
        
    plt.show()

if __name__ == '__main__':
    main()
