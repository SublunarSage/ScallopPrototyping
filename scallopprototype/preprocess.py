# preprocess.py
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def preprocess_image(image_path, target_size=(28, 28), visualize=False):
    """
    Preprocess a handwriting image for recognition.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for the processed image
        visualize (bool): If True, displays the processing steps
        
    Returns:
        numpy.ndarray: Preprocessed image as a normalized array
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to create binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Resize to target size
    resized = cv2.resize(binary, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized = resized / 255.0
    
    # Visualize preprocessing steps if requested
    if visualize:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(142)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        plt.subplot(143)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary')
        plt.axis('off')
        
        plt.subplot(144)
        plt.imshow(normalized, cmap='gray')
        plt.title('Normalized')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return normalized

def test_preprocessing():
    """
    Simple test function for the preprocessing pipeline.
    """
    image_path = input("Enter path to a test image: ")
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return
    
    try:
        processed = preprocess_image(image_path, visualize=True)
        print(f"Preprocessed image shape: {processed.shape}")
        print(f"Pixel value range: {processed.min()} to {processed.max()}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    test_preprocessing()

