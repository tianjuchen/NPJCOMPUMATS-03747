import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def images_to_dataset(image_dir, output_path, img_size=(128, 128), normalize=True, test_split=0.2):
    """
    Convert images in a directory to a dataset format compatible with the AE model.
    
    Parameters:
    - image_dir: Path to directory containing images
    - output_path: Path where to save the .npz file
    - img_size: Target image size (height, width)
    - normalize: Whether to normalize images to [0, 1]
    - test_split: Fraction of data to use for test set
    """
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = []
    for file in os.listdir(image_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            image_files.append(os.path.join(image_dir, file))
    
    if not image_files:
        raise ValueError(f"No valid image files found in {image_dir}")
    
    print(f"Found {len(image_files)} image files")
    
    # Load and preprocess images
    images = []
    failed_files = []
    
    for i, file_path in enumerate(image_files):
        try:
            # Method 1: Using OpenCV (if available)
            try:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("OpenCV failed to read image")
            except:
                # Method 2: Using PIL as fallback
                img = np.array(Image.open(file_path).convert('L'))
            
            # Resize image
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            
            # Normalize if requested
            if normalize:
                img = img.astype(np.float32)
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
            else:
                img = img.astype(np.float32)
            
            images.append(img)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            failed_files.append(file_path)
    
    if failed_files:
        print(f"Failed to process {len(failed_files)} files")
    
    if not images:
        raise ValueError("No images were successfully processed")
    
    # Convert to numpy array
    images_array = np.array(images)
    print(f"Final image array shape: {images_array.shape}")
    
    # Split into train and test
    if test_split > 0:
        X_train, X_test = train_test_split(images_array, test_size=test_split, random_state=42)
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Save both train and test
        np.savez(output_path, X_train=X_train, X_test=X_test)
        print(f"Saved dataset to {output_path} with train and test splits")
    else:
        # Save all as training data
        np.savez(output_path, X_train=images_array, X_test=np.array([]))
        print(f"Saved dataset to {output_path} (train only)")
    
    return images_array

def visualize_sample(images, num_samples=5):
    """Visualize sample images from the dataset"""
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        if i < len(images):
            axes[i].imshow(images[i], cmap='gray')
            axes[i].set_title(f'Sample {i+1}')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def create_dataset_from_images():
    """Main function to create dataset from images"""
    
    # Configuration
    image_directory = "./images"  # Change this to your image directory
    output_file = "./data/image_dataset_128_128.npz"
    img_size = (128, 128)  # Should match your AE model input size
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Convert images to dataset
        images = images_to_dataset(
            image_dir=image_directory,
            output_path=output_file,
            img_size=img_size,
            normalize=True,
            test_split=0.2
        )
        
        # Visualize some samples
        print("Visualizing sample images...")
        visualize_sample(images)
        
        # Load and verify the saved dataset
        print("\nVerifying saved dataset...")
        data = np.load(output_file)
        
        if 'X_train' in data:
            print(f"Train set shape: {data['X_train'].shape}")
        if 'X_test' in data:
            print(f"Test set shape: {data['X_test'].shape}")
        
        print("Dataset creation completed successfully!")
        
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")

def process_single_image_for_inference(image_path, img_size=(128, 128)):
    """
    Process a single image for inference with the AE model.
    Returns the image in the format expected by the model.
    """
    try:
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.array(Image.open(image_path).convert('L'))
        
        # Resize
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        
        # Add batch and channel dimensions
        img_processed = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
        
        print(f"Processed image shape: {img_processed.shape}")
        return img_processed
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import cv2
        from PIL import Image
    except ImportError:
        print("Please install required packages:")
        print("pip install opencv-python pillow scikit-learn")
        exit(1)
    
    create_dataset_from_images()
    
    # Example for processing a single image
    # single_image_path = "./sample_image.jpg"
    # processed_image = process_single_image_for_inference(single_image_path)