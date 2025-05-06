import os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from mnistiseven import MNISTParityNet, mnist_parity_loader

def load_model_from_safetensor(model_path, k=3, provenance="difftopkproofs"):
    """
    Load a model from a safetensor file
    
    Args:
        model_path: Path to the safetensor file
        k: Top-k parameter for Scallop
        provenance: Provenance type for Scallop
    
    Returns:
        The loaded model
    """
    # Create an empty model with the same architecture
    model = MNISTParityNet(provenance=provenance, k=k)
    
    # Load the model weights from the safetensor file
    state_dict = load_file(model_path)
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    
    # Put model in eval mode
    model.eval()
    
    return model

def test_model_on_digit(model, digit, expected_output):
    """
    Test the model on a specific digit
    
    Args:
        model: The model to test
        digit: The digit to test (0-9)
        expected_output: The expected output (0 for odd, 1 for even)
    
    Returns:
        True if the model predicts correctly, False otherwise
    """
    # Create a one-hot encoding for the digit
    digit_tensor = torch.zeros(1, 10)
    digit_tensor[0, digit] = 1.0  # Full certainty in the digit
    
    # Run the model on the digit
    with torch.no_grad():
        output = model.parity_detector(digit=digit_tensor)
    
    # Get the predicted output
    predicted = output.argmax(dim=1).item()
    
    # Check if the prediction is correct
    is_correct = predicted == expected_output
    
    print(f"Digit: {digit}, Expected: {'even' if expected_output == 1 else 'odd'}, "
          f"Predicted: {'even' if predicted == 1 else 'odd'} - "
          f"{'✓' if is_correct else '✗'}")
    
    return is_correct

def test_model_on_test_set(model, test_loader, max_samples=100):
    """
    Test the model on the test set
    
    Args:
        model: The model to test
        test_loader: DataLoader for the test set
        max_samples: Maximum number of samples to test
    
    Returns:
        Accuracy on the test set
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            if total >= max_samples:
                break
                
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f"\nTest set accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return accuracy

def verify_scallop_rules(model):
    """
    Verify that the Scallop rules are working correctly by testing on all digits
    
    Args:
        model: The model to test
    
    Returns:
        True if all rules are working correctly, False otherwise
    """
    print("\nVerifying Scallop rules...")
    all_correct = True
    
    # Test even digits (0, 2, 4, 6, 8)
    for digit in [0, 2, 4, 6, 8]:
        is_correct = test_model_on_digit(model, digit, 1)  # 1 for even
        all_correct = all_correct and is_correct
    
    # Test odd digits (1, 3, 5, 7, 9)
    for digit in [1, 3, 5, 7, 9]:
        is_correct = test_model_on_digit(model, digit, 0)  # 0 for odd
        all_correct = all_correct and is_correct
    
    if all_correct:
        print("\nAll Scallop rules are working correctly! ✓")
    else:
        print("\nSome Scallop rules are not working correctly. ✗")
    
    return all_correct

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("test_safetensor_model")
    parser.add_argument("--model-path", type=str, default="../model/mnist_iseven_model.safetensors", 
                        help="Path to the safetensor model file")
    parser.add_argument("--top-k", type=int, default=3, 
                        help="Top-k parameter for Scallop")
    parser.add_argument("--provenance", type=str, default="difftopkproofs", 
                        help="Provenance type for Scallop")
    parser.add_argument("--test-full", action="store_true", 
                        help="Run a full test on the test set")
    parser.add_argument("--max-samples", type=int, default=100, 
                        help="Maximum number of samples to test")
    
    args = parser.parse_args()
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = load_model_from_safetensor(
        model_path=args.model_path,
        k=args.top_k,
        provenance=args.provenance
    )
    
    # Verify Scallop rules
    rules_correct = verify_scallop_rules(model)
    
    # Optionally test on the full test set
    if args.test_full:
        print("\nRunning full test on test set...")
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
        _, test_loader = mnist_parity_loader(data_dir, batch_size_train=64, batch_size_test=64)
        test_model_on_test_set(model, test_loader, args.max_samples)