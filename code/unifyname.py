import sys
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def images_to_dataset(image_dir, scale="1um"):
    """
    Convert images in a directory to a dataset format compatible with the AE model.

    Parameters:
    - image_dir: Path to directory containing images
    - scale: Scale prefix to add to filenames
    """

    # Supported image extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # Find all image files
    image_files = []
    for file in os.listdir(image_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            image_files.append(os.path.join(image_dir, file))

    if not image_files:
        raise ValueError(f"No valid image files found in {image_dir}")

    print(f"Found {len(image_files)} image files")

    # Rename files with scale prefix
    renamed_files = []
    for i, file_path in enumerate(image_files):
        # Get directory and filename components
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        
        # Create new filename
        new_filename = f"{scale}_{i+1}{ext}"
        new_file_path = os.path.join(directory, new_filename)
        
        print(f"Renaming: {filename} -> {new_filename}")
        
        # Rename the file
        os.rename(file_path, new_file_path)
        renamed_files.append(new_file_path)

    return renamed_files


if __name__ == "__main__":
    image_directory = "./../superalloy/gamma/"
    scale = "no_scale"

    new_files = images_to_dataset(os.path.join(image_directory, scale), scale=scale)
    print("Renamed files:", new_files)