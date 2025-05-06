# evaluate.py
import os
import numpy as np
import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from recognition import recognize_handwriting

def evaluate_model(model_path, test_samples, test_labels, visualize=False):
    """
    Evaluate the model on test data.
    
    Args:
        model_path (str): Path to the trained model
        test_samples (numpy.ndarray): Test samples
        test_labels (numpy.ndarray): Test labels
        visualize (bool): If True, displays a confusion matrix
        
    Returns:
        float: Accuracy of the model
        list: Predictions for each test sample
    """
    predictions = []
    
    print("Evaluating model on test samples...")
    for i, sample in enumerate(test_samples):
        # Save sample as temporary image
        temp_image = sample.reshape(28, 28) * 255
        temp_path = "temp_eval_image.png"
        cv2.imwrite(temp_path, temp_image)
        
        # Recognize handwriting
        try:
            prediction = recognize_handwriting(model_path, temp_path)
            predictions.append(prediction)
        except Exception as e:
            print(f"Error recognizing sample {i}: {e}")
            predictions.append("Unknown")
        
        # Print progress
        if (i + 1) % 10 == 0 or i == len(test_samples) - 1:
            print(f"Processed {i + 1}/{len(test_samples)} test samples")
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    # Visualize results if requested
    if visualize:
        # Create confusion matrix
        unique_labels = sorted(set(list(test_labels) + predictions))
        cm = confusion_matrix(test_labels, predictions, labels=unique_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=unique_labels, yticklabels=unique_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
        plt.tight_layout()
        plt.show()
        
        # Display some misclassified examples
        misclassified_indices = [i for i, (y, p) in enumerate(zip(test_labels, predictions)) if y != p]
        
        if misclassified_indices:
            num_examples = min(10, len(misclassified_indices))
            plt.figure(figsize=(12, 2 * (num_examples // 5 + 1)))
            
            for i, idx in enumerate(misclassified_indices[:num_examples]):
                plt.subplot(num_examples // 5 + 1, 5, i + 1)
                plt.imshow(test_samples[idx].reshape(28, 28), cmap='gray')
                plt.title(f'True: {test_labels[idx]}\nPred: {predictions[idx]}')
                plt.axis('off')
                
            plt.tight_layout()
            plt.show()
    
    # Remove temporary file
    if os.path.exists("temp_eval_image.png"):
        os.remove("temp_eval_image.png")
    
    return accuracy, predictions

def test_evaluation():
    """
    Load test data and evaluate a trained model.
    This is a simplified function since we would normally use the dataset module.
    """
    from dataset import create_dataset, split_dataset
    
    data_dir = input("Enter directory containing image data: ")
    model_path = input("Enter path to trained model: ")
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found at {data_dir}")
        return
        
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    
    try:
        # Create and split dataset
        samples, labels, _ = create_dataset(data_dir)
        X_train, X_test, y_train, y_test = split_dataset(samples, labels)
        
        # Evaluate model
        accuracy, predictions = evaluate_model(model_path, X_test, y_test, visualize=True)
        print(f"Model accuracy: {accuracy:.2f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    test_evaluation()