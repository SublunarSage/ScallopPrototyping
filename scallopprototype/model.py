# model.py
import os
import numpy as np
import pickle
from scallopy import ScallopContext

def create_scallop_model():
    """
    Create and configure a Scallop context with rules for handwriting recognition.
    
    Returns:
        ScallopContext: Configured Scallop context
    """
    # Create a Scallop context
    ctx = ScallopContext()
    
    # Define the relation types with proper types
    ctx.add_relation("Pixel", (int, int, float))  # usize, usize, float
    ctx.add_relation("Label", (int, str))         # usize, String
    
    # Define rules for handwriting recognition
    scallop_program = """
    type Pixel(id: usize, position: usize, value: float)
    type Label(id: usize, value: String)
    
    // Feature extraction
    type RegionAverage(id: usize, region: usize, avg: float)
    rel RegionAverage(id, region, avg) :-
        // Divide the image into regions and calculate average pixel values
        Pixel(id, position, value),
        region = position / 49,
        // Divide 28x28 image into 16 regions
        agg<<avg = mean(value)>> by (id, region).
    
    // Classification rule using probabilistic reasoning
    type CharacterFeature(id: usize, feature_vector: Vec<float>)
    rel CharacterFeature(id, features) :-
        // Collect features from all regions
        features = vec_agg<<region_avg>> by (id, region) <-
            RegionAverage(id, region, region_avg),
            ord by (region).
    
    // Probabilistic label prediction
    type PredictedLabel(id: usize, label: String)
    rel PredictedLabel(id, label) :-
        CharacterFeature(id, features),
        Label(other_id, label),
        // Compute similarity to known examples
        other_id != id,
        // This would be enhanced with more sophisticated
        // similarity metrics in a real application
        id @ label.  // Probabilistic assignment
    """
    
    ctx.add_program(scallop_program)
    return ctx

def train_model(samples, labels, model_path="handwriting_model.pkl"):
    """
    Train a Scallop model for handwriting recognition.
    
    Args:
        samples (numpy.ndarray): Training samples
        labels (numpy.ndarray): Training labels
        model_path (str): Path to save the trained model
        
    Returns:
        ScallopContext: Trained model
    """
    ctx = create_scallop_model()
    
    # Convert samples and labels to Scallop facts
    pixel_facts = []
    label_facts = []
    
    print("Converting samples to Scallop facts...")
    for i, (sample, label) in enumerate(zip(samples, labels)):
        for j, pixel_value in enumerate(sample):
            pixel_facts.append(("Pixel", (i, j, float(pixel_value))))
        label_facts.append(("Label", (i, label)))
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(samples)} samples")
    
    # Add facts to the context
    print("Adding facts to Scallop context...")
    ctx.add_facts(pixel_facts)
    ctx.add_facts(label_facts)
    
    # Save the model
    print(f"Saving model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(ctx, f)
    
    return ctx

def load_model(model_path="handwriting_model.pkl"):
    """
    Load a trained Scallop model.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        ScallopContext: Loaded model
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found at {model_path}")
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    return model

def test_model_creation():
    """
    Test function for model creation.
    """
    try:
        model = create_scallop_model()
        print("Scallop model created successfully!")
        print(f"Model has the following relations:")
        for rel_type in model.relations():
            print(f"  - {rel_type}")
    except Exception as e:
        print(f"Error creating Scallop model: {e}")

if __name__ == "__main__":
    test_model_creation()