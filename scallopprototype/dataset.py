import os
import numpy as np
from preprocess import preprocess_image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def create_dataset(data_dir, labels_file="labels.txt", visualize=False):
    """
    Create a dataset from processed handwriting samples.
    
    Args:
        data_dir (str): Directory containing the image files and labels
        labels_file (str): Name of the file mapping images to labels
        visualize (bool): If True, displays some sample images
        
    Returns:
        tuple: (samples, labels) as numpy arrays
    """
    # Read labels
    samples = []
    labels = []
    filenames = []
    
    labels_path = os.path.join(data_dir, labels_file)
    
    if not os.path.exists(labels_path):
        raise ValueError(f"Labels file not found at {labels_path}")
    
    with open(labels_path, 'r') as f:
        for line in f:
            if ',' not in line:
                continue
                
            filename, label = line.strip().split(',')
            
            # Preprocess image
            image_path = os.path.join(data_dir, filename)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found at {image_path}, skipping")
                continue
                
            try:
                processed_image = preprocess_image(image_path)
                
                # Add to dataset
                samples.append(processed_image.flatten())
                labels.append(label)
                filenames.append(filename)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    
    if not samples:
        raise ValueError("No valid samples were processed")
    
    # Visualize some samples if requested
    if visualize and samples:
        plt.figure(figsize=(10, 5))
        for i in range(min(10, len(samples))):
            plt.subplot(2, 5, i+1)
            plt.imshow(samples[i].reshape(28, 28), cmap='gray')
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return np.array(samples), np.array(labels), filenames

def split_dataset(samples, labels, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.
    
    Args:
        samples (numpy.ndarray): Sample images
        labels (numpy.ndarray): Corresponding labels
        test_size (float): Proportion of samples to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(samples, labels, test_size=test_size, random_state=random_state)

def test_dataset_creation():
    """
    Test function for dataset creation.
    """
    data_dir = input("Enter the directory containing your images and labels.txt: ")
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return
    
    try:
        samples, labels, filenames = create_dataset(data_dir, visualize=True)
        print(f"Dataset created with {len(samples)} samples")
        print(f"Sample shape: {samples[0].shape}")
        
        # Count samples per label
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\nSamples per label:")
        for label, count in zip(unique_labels, counts):
            print(f"  {label}: {count}")
        
        # Split dataset
        X_train, X_test, y_train, y_test = split_dataset(samples, labels)
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")

if __name__ == "__main__":
    test_dataset_creation()