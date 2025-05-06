# main.py
import os
import argparse
from dataset import create_dataset, split_dataset
from model import train_model
from recognition import recognize_handwriting
from evaluate import evaluate_model

def run_pipeline(data_dir, model_path="handwriting_model.pkl", test_image=None):
    """
    Run the complete handwriting recognition pipeline.
    
    Args:
        data_dir (str): Directory containing image data and labels.txt
        model_path (str): Path to save/load the model
        test_image (str, optional): Path to a test image for recognition
    """
    # Step 1: Create dataset
    print("\n=== Creating dataset ===")
    samples, labels, _ = create_dataset(data_dir)
    print(f"Dataset created with {len(samples)} samples")
    
    # Step 2: Split dataset
    print("\n=== Splitting dataset ===")
    X_train, X_test, y_train, y_test = split_dataset(samples, labels)
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # Step 3: Train model
    print("\n=== Training model ===")
    model = train_model(X_train, y_train, model_path)
    print(f"Model trained and saved to {model_path}")
    
    # Step 4: Evaluate model
    print("\n=== Evaluating model ===")
    accuracy, _ = evaluate_model(model_path, X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Step 5: Test recognition (if a test image is provided)
    if test_image:
        if os.path.exists(test_image):
            print("\n=== Testing recognition ===")
            result = recognize_handwriting(model_path, test_image)
            print(f"Recognized character: {result}")
        else:
            print(f"Test image not found at {test_image}")

def parse_args():
    parser = argparse.ArgumentParser(description="Handwriting Recognition with Scallop")
    parser.add_argument("--data_dir", required=True, help="Directory containing images and labels.txt")
    parser.add_argument("--model_path", default="handwriting_model.pkl", help="Path to save/load model")
    parser.add_argument("--test_image", help="Path to a test image for recognition")
    parser.add_argument("--mode", choices=["full", "preprocess", "train", "evaluate", "recognize"], 
                        default="full", help="Mode to run")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Data directory not found at {args.data_dir}")
        exit(1)
    
    if args.mode == "full":
        run_pipeline(args.data_dir, args.model_path, args.test_image)
    
    elif args.mode == "preprocess":
        from preprocess import test_preprocessing
        test_preprocessing()
    
    elif args.mode == "train":
        # Create dataset and train model
        samples, labels, _ = create_dataset(args.data_dir)
        X_train, X_test, y_train, y_test = split_dataset(samples, labels)
        train_model(X_train, y_train, args.model_path)
        print(f"Model trained and saved to {args.model_path}")
    
    elif args.mode == "evaluate":
        # Check if model exists
        if not os.path.exists(args.model_path):
            print(f"Model file not found at {args.model_path}")
            exit(1)
            
        # Create dataset and evaluate model
        samples, labels, _ = create_dataset(args.data_dir)
        X_train, X_test, y_train, y_test = split_dataset(samples, labels)
        accuracy, _ = evaluate_model(args.model_path, X_test, y_test, visualize=True)
        print(f"Model accuracy: {accuracy:.2f}")
    
    elif args.mode == "recognize":
        # Check if model and test image exist
        if not os.path.exists(args.model_path):
            print(f"Model file not found at {args.model_path}")
            exit(1)
            
        if not args.test_image or not os.path.exists(args.test_image):
            print("Please provide a valid test image path")
            exit(1)
            
        # Recognize handwriting
        result = recognize_handwriting(args.model_path, args.test_image, visualize=True)
        print(f"Recognized character: {result}")