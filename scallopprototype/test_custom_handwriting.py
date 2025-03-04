#!/usr/bin/env python
import os
import sys
import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from ugh import AlphabetWithScallop

def load_model(model_path):
    """Load a saved model from safetensors format"""
    # Initialize the model
    model = AlphabetWithScallop()
    
    # Load weights from safetensors file
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    
    # Set model to evaluation mode
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess a custom handwriting image to match EMNIST format"""
    # Open image and convert to grayscale
    image = Image.open(image_path).convert('L')
    
    # EMNIST images are 28x28 pixels
    image = image.resize((28, 28))
    
    # Apply same transformations as used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # EMNIST images are flipped and rotated 90 degrees
    # We need to flip and rotate our input to match
    image = transforms.functional.rotate(image, -90)
    image = transforms.functional.hflip(image)
    
    # Transform and add batch dimension
    tensor = transform(image).unsqueeze(0)
    return tensor

def predict(model, image_tensor):
    """Run inference on preprocessed image"""
    with torch.no_grad():
        output = model(image_tensor)
        # Get the letter with highest probability
        # Add 1 to convert from 0-indexed to 1-indexed (EMNIST format)
        letter_idx = output.argmax(1).item() + 1
        # Convert to letter (1=A, 2=B, etc.)
        letter = chr(ord('A') + letter_idx - 1)
        # Get confidence level
        confidence = output[0, letter_idx-1].item()
    return letter, confidence

def display_results(image_path, letter, confidence):
    """Display the input image and prediction results"""
    # Display the original image
    image = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {letter} (Confidence: {confidence:.2f})")
    plt.axis('off')
    plt.show()

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_custom_handwriting.py <model_path> <image_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        sys.exit(1)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    print(f"Processing image {image_path}...")
    image_tensor = preprocess_image(image_path)
    
    print("Running inference...")
    letter, confidence = predict(model, image_tensor)
    
    print(f"Prediction: {letter} with confidence {confidence:.4f}")
    display_results(image_path, letter, confidence)
    
    # Also print top 3 predictions with confidence values
    with torch.no_grad():
        output = model(image_tensor)
        values, indices = torch.topk(output, 3, dim=1)
        
    print("\nTop 3 predictions:")
    for i in range(3):
        idx = indices[0, i].item()
        val = values[0, i].item()
        pred_letter = chr(ord('A') + idx)
        print(f"{pred_letter}: {val:.4f}")

if __name__ == "__main__":
    main()