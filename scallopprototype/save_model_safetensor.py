import os
import torch
from safetensors.torch import save_file
from mnistiseven import MNISTParityNet, mnist_parity_loader, Trainer

def train_and_save_model(save_path, n_epochs=1, k=3, provenance="difftopkproofs", learning_rate=0.001, loss_fn="bce"):
    """
    Train the MNISTParityNet model and save it as a safetensor file
    
    Args:
        save_path: Path where to save the model
        n_epochs: Number of epochs to train
        k: Top-k parameter for Scallop
        provenance: Provenance type for Scallop
        learning_rate: Learning rate for training
        loss_fn: Loss function to use ("bce" or "nll")
    """
    # Set random seeds for reproducibility
    torch.manual_seed(1234)
    
    # Data directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    
    # Create data loaders
    train_loader, test_loader = mnist_parity_loader(data_dir, batch_size_train=64, batch_size_test=64)
    
    # Create and train the model
    trainer = Trainer(train_loader, test_loader, learning_rate, loss_fn, k, provenance)
    
    print(f"Training model for {n_epochs} epochs...")
    trainer.train(n_epochs)
    
    # Get the trained model
    model = trainer.network
    
    # Put model in eval mode
    model.eval()
    
    # Get model state dictionary
    state_dict = model.state_dict()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the model as safetensor file
    save_file(state_dict, save_path)
    print(f"Model saved to {save_path}")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("save_model_safetensor")
    parser.add_argument("--save-path", type=str, default="../model/mnist_iseven_model.safetensors", 
                        help="Path where to save the model")
    parser.add_argument("--n-epochs", type=int, default=1, 
                        help="Number of epochs to train")
    parser.add_argument("--top-k", type=int, default=3, 
                        help="Top-k parameter for Scallop")
    parser.add_argument("--provenance", type=str, default="difftopkproofs", 
                        help="Provenance type for Scallop")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                        help="Learning rate for training")
    parser.add_argument("--loss-fn", type=str, default="bce", 
                        help="Loss function to use (bce or nll)")
    
    args = parser.parse_args()
    
    train_and_save_model(
        save_path=args.save_path,
        n_epochs=args.n_epochs,
        k=args.top_k,
        provenance=args.provenance,
        learning_rate=args.learning_rate,
        loss_fn=args.loss_fn
    )