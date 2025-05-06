# recognition.py
import cv2
import numpy as np
import os
from preprocess import preprocess_image
from model import load_model
import matplotlib.pyplot as plt

def recognize_handwriting(model_path, image_path, visualize=False):
    """
    Recognize handwriting in an image using a trained Scallop model.
    
    Args:
        model_path (str): Path to the trained model
        image_path (str): Path to the image to recognize
        visualize (bool): If True, displays the image being recognized
        
    Returns:
        str: Recognized character
    """
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    if visualize:
        plt.figure(figsize=(4, 4))
        plt.imshow(processed_image, cmap='gray')
        plt.title('Image being recognized')
        plt.axis('off')
        plt.show()
    
    # Convert to Scallop facts - using a unique ID (999) for the test image
    pixel_facts = [("Pixel", (999, j, float(pixel_value))) for j, pixel_value in enumerate(processed_image.flatten())]
    
    # Add facts to the model
    model.add_facts(pixel_facts)
    
    # Run the model
    result = model.run("PredictedLabel")
    
    # Get the predicted label
    if result and len(result) > 0:
        # Sort by probability if multiple labels returned
        sorted_results = sorted(result, key=lambda x: x[0], reverse=True)
        predicted_label = sorted_results[0][1]
        return predicted_label
    else:
        return "Unknown"

def test_recognition():
    """
    Test function for handwriting recognition.
    """
    model_path = input("Enter path to trained model file: ")
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    
    image_path = input("Enter path to image to recognize: ")
    if not os.path.exists(image_path):
        print(f"Image file not found at {image_path}")
        return
    
    try:
        result = recognize_handwriting(model_path, image_path, visualize=True)
        print(f"Recognized character: {result}")
    except Exception as e:
        print(f"Error during recognition: {e}")

if __name__ == "__main__":
    test_recognition()