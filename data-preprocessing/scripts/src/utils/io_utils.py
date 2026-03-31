import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_nifti_image(image_path):
    img = nib.load(image_path)
    return img.get_fdata()

def save_np_to_nifti(array, filepath, affine=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if affine is None:
        affine = np.eye(4)
    nifti_img = nib.Nifti1Image(array.astype(np.float32), affine)
    nib.save(nifti_img, filepath)

def save_png(image, path, cmap="bone"):
    """Saves an array as a PNG."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if cmap == "binary":
        img_data = (image > 0).astype(np.uint8) * 255
        Image.fromarray(img_data, mode="L").save(path)
    else:
        # Scale to 0-1 for plt.imsave if not already
        plt.imsave(path, image, cmap=cmap)

def get_ids_from_ungood_test_folder(output_dir):
    """Infers unique patient IDs from a folder of slices."""
    img_dir = os.path.join(output_dir, "test", "Ungood", "img")
    if not os.path.isdir(img_dir):
        return set()
    
    ids = set()
    for fname in os.listdir(img_dir):
        if not fname.endswith((".nii", ".nii.gz", ".png")):
            continue
        # Remove extension and the slice suffix (everything after last underscore)
        stem = fname.split('.')[0]
        pid = "_".join(stem.split("_")[:-1])
        if pid: ids.add(pid)
    return ids