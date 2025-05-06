import scallopy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Config
MNIST_DATA_PATH = "./data"
SCALLOP_PROGRAM_PATH = "./scallop_bitwise/bitwise.scl"
THRESHOLD = 128
IMG_SIZE = 28
VECTOR_LENGTH = IMG_SIZE * IMG_SIZE

# Helper Functions
def load_mnist(data_path, train=True, batch_size=64):
    """Loads the MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)) # Normalization not needed for thresholding
    ])
    dataset = torchvision.datasets.MNIST(
        root=data_path, train=train, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return dataloader

def image_to_boolean_vector(image_tensor, threshold=THRESHOLD):
    """Converts a single MNIST image tensor to a boolean vector."""
    # Ensure image is 2D (H, W) or (1, H, W)
    if image_tensor.ndim == 3 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)
    elif image_tensor.ndim != 2:
        raise ValueError(f"Unexpected image dimensions: {image_tensor.shape}")

    # Flatten and apply threshold
    flat_tensor = image_tensor.view(-1) # Flatten to 1D
    boolean_vector = (flat_tensor * 255 > threshold).tolist() # Scale back to 0-255 and compare
    return boolean_vector

def format_vector_for_scallop(vector_id, bool_vector):
    """Formats a boolean vector into Scallop facts."""
    # Expects bool_vector to be a list of booleans
    return [(vector_id, i, val) for i, val in enumerate(bool_vector)]

def reconstruct_boolean_vector(scallop_result_tuples, expected_length=VECTOR_LENGTH, op_name="Operation"):
    """Reconstructs a boolean vector from Scallop query results."""
    if not scallop_result_tuples:
        print(f"Warning [{op_name}]: No tuples found for reconstruction.")
        return []

    # Check for unexpected tuple structure early
    if scallop_result_tuples and len(scallop_result_tuples[0]) != 3:
         print(f"Error [{op_name}]: Expected result tuples of length 3 (id, index, value), but got length {len(scallop_result_tuples[0])}.")
         return [] # Cannot proceed

    # Sort by index to ensure correct order
    # Handle potential non-integer indices gracefully during sort
    try:
        sorted_result = sorted(scallop_result_tuples, key=lambda item: int(item[1]))
    except (ValueError, TypeError, IndexError) as e:
        print(f"Error [{op_name}]: Could not sort result tuples by index. Invalid data? Error: {e}")
        # print("Sample tuples:", scallop_result_tuples[:5]) # Debug: show sample data
        return []


    num_results = len(sorted_result)
    print(f"Debug [{op_name}]: Received {num_results} tuples after retrieval.") # Debug print

    # Explicitly check for duplicates based on index
    indices_present = [item[1] for item in sorted_result]
    if len(indices_present) != len(set(indices_present)):
        print(f"Warning [{op_name}]: Duplicate indices found in results! Count from Scallop: {num_results}, Unique indices: {len(set(indices_present))}")
        # Decide how to handle duplicates - here we might take the first occurrence after sorting
        # A more robust way might involve grouping by index and deciding (e.g., error, take first)
        unique_results = []
        seen_indices = set()
        for item in sorted_result:
            if item[1] not in seen_indices:
                unique_results.append(item)
                seen_indices.add(item[1])
        print(f"Debug [{op_name}]: Filtered to {len(unique_results)} tuples after removing duplicates.")
        sorted_result = unique_results # Use the de-duplicated list
        num_results = len(sorted_result)


    # Check for completeness against expected length
    if num_results != expected_length:
        print(f"Warning [{op_name}]: Reconstructed vector length mismatch. Got {num_results} unique tuples, expected {expected_length}.")
        indices_present_set = {item[1] for item in sorted_result}
        expected_indices = set(range(expected_length))
        missing = expected_indices - indices_present_set
        extra = indices_present_set - expected_indices
        if missing:
            print(f"  Missing indices count: {len(missing)}. Examples: {list(missing)[:5]}")
        if extra:
            print(f"  Extra indices count: {len(extra)}. Examples: {list(extra)[:5]}")
        # Decide on handling: padding/truncating or error. Padding is often safer for visualization.
    elif num_results == expected_length:
         print(f"Info [{op_name}]: Successfully received expected number of results ({expected_length}).")


    # Extract boolean values in order
    bool_vector = [item[2] for item in sorted_result]

    # Pad or truncate if necessary to match expected length for visualization
    if len(bool_vector) < expected_length:
        bool_vector.extend([False] * (expected_length - len(bool_vector)))
    elif len(bool_vector) > expected_length:
        bool_vector = bool_vector[:expected_length]

    return bool_vector

def visualize_boolean_vector(bool_vector, title="Boolean Image", img_size=IMG_SIZE):
    """Visualizes a boolean vector as a 2D image."""
    expected_elements = img_size * img_size
    if not bool_vector: # Handle empty vector case
        print(f"Error visualizing '{title}': Input vector is empty.")
        img_array = np.zeros((img_size, img_size)) # Show black image
    elif len(bool_vector) != expected_elements:
        print(f"Warning visualizing '{title}': Vector length {len(bool_vector)} doesn't match expected {expected_elements}. Reshaping might fail or be inaccurate.")
        # Attempt to reshape anyway, might error out or produce weird results
        try:
             img_array = np.array(bool_vector, dtype=float).reshape((img_size, img_size))
        except ValueError as e:
             print(f"  Reshape failed: {e}. Displaying empty image.")
             img_array = np.zeros((img_size, img_size)) # Show black image as fallback
    else:
         img_array = np.array(bool_vector, dtype=float).reshape((img_size, img_size))

    plt.figure(figsize=(4, 4))
    plt.imshow(img_array, cmap='gray_r', vmin=0, vmax=1) # white=False(0), black=True(1)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    # 1. Check and Load Scallop program string (Load only once)
    if not os.path.exists(SCALLOP_PROGRAM_PATH):
        print(f"Error: Scallop program file '{SCALLOP_PROGRAM_PATH}' not found.")
        exit()
    try:
        with open(SCALLOP_PROGRAM_PATH, "r") as f:
            scallop_program_str = f.read()
        print(f"Scallop program loaded successfully from '{SCALLOP_PROGRAM_PATH}'.")
    except Exception as e:
        print(f"Error reading Scallop program file: {e}")
        exit()

    # 2. Load MNIST Data (Load only once)
    print("Loading MNIST data...")
    # (Error handling for data loading as before)
    mnist_loader = load_mnist(MNIST_DATA_PATH, batch_size=2)
    images, labels = next(iter(mnist_loader))
    img_a_tensor = images[0]
    img_b_tensor = images[1]
    label_a = labels[0].item()
    label_b = labels[1].item()
    print(f"Loaded two MNIST images. Image A: Label {label_a}, Image B: Label {label_b}")


    # 3. Preprocess Images to Boolean Vectors (Preprocess only once)
    bool_vec_a = image_to_boolean_vector(img_a_tensor)
    bool_vec_b = image_to_boolean_vector(img_b_tensor)
    print(f"Converted images to boolean vectors (length {len(bool_vec_a)}).")


    # Visualize original thresholded images
    visualize_boolean_vector(bool_vec_a, title=f"Input A (Label: {label_a}) Thresholded")
    visualize_boolean_vector(bool_vec_b, title=f"Input B (Label: {label_b}) Thresholded")

    # 4. Define Vector IDs
    id_a = 0
    id_b = 1
    id_not_a = 10
    id_or_ab = 11
    id_and_ab = 12
    id_xor_ab = 13
    
    # 5. Prepare Base Input Facts (Format only once)
    facts_a = format_vector_for_scallop(id_a, bool_vec_a)
    facts_b = format_vector_for_scallop(id_b, bool_vec_b)
    print(f"Base facts prepared for vector {id_a} and vector {id_b}.")


    # --- Perform Operations using Scallop ---
    # *** RECREATE CONTEXT FOR EACH OPERATION ***

    # -- Example 1: NOT Operation on Vector A --
    print("\n--- Performing NOT operation on Vector A ---")
    op_name_not = "NOT"
    try:
        ctx_not = scallopy.ScallopContext(provenance="unit")
        ctx_not.add_program(scallop_program_str) # Add program to this context
        ctx_not.add_facts("vector_element", facts_a) # Add necessary base facts
        # Only add the request for this specific operation
        request_not_fact = [(id_a, id_not_a)]
        ctx_not.add_facts("requested_not_op", request_not_fact)
        print(f"[{op_name_not}] Added request: requested_not_op({id_a}, {id_not_a})")

        start_time = time.time()
        ctx_not.run() # Compute
        end_time = time.time()
        print(f"[{op_name_not}] Scallop run executed in {end_time - start_time:.4f} seconds.")

        scallop_not_result_tuples = list(ctx_not.relation("vector_not_result"))
        print(f"[{op_name_not}] Retrieved {len(scallop_not_result_tuples)} raw result tuples.")

        # No need to remove facts, context will be discarded

        if scallop_not_result_tuples:
            bool_vec_not_a = reconstruct_boolean_vector(scallop_not_result_tuples, op_name=op_name_not)
            visualize_boolean_vector(bool_vec_not_a, title=f"Result: NOT A (Vector {id_not_a})")
        else:
            print(f"[{op_name_not}] No result found for NOT operation.")
    except Exception as e:
        print(f"Error during {op_name_not} operation: {e}")
    # Context ctx_not goes out of scope here

    # -- Example 2: OR Operation on Vector A and Vector B --
    print("\n--- Performing OR operation on Vector A and B ---")
    op_name_or = "OR"
    try:
        ctx_or = scallopy.ScallopContext(provenance="unit")
        ctx_or.add_program(scallop_program_str)
        ctx_or.add_facts("vector_element", facts_a) # Need facts for A
        ctx_or.add_facts("vector_element", facts_b) # Need facts for B
        request_or_fact = [(id_a, id_b, id_or_ab)]
        ctx_or.add_facts("requested_or_op", request_or_fact)
        print(f"[{op_name_or}] Added request: requested_or_op({id_a}, {id_b}, {id_or_ab})")

        start_time = time.time()
        ctx_or.run()
        end_time = time.time()
        print(f"[{op_name_or}] Scallop run executed in {end_time - start_time:.4f} seconds.")

        scallop_or_result_tuples = list(ctx_or.relation("vector_or_result"))
        print(f"[{op_name_or}] Retrieved {len(scallop_or_result_tuples)} raw result tuples.")

        if scallop_or_result_tuples:
            bool_vec_or_ab = reconstruct_boolean_vector(scallop_or_result_tuples, op_name=op_name_or)
            visualize_boolean_vector(bool_vec_or_ab, title=f"Result: A OR B (Vector {id_or_ab})")
        else:
            print(f"[{op_name_or}] No result found for OR operation.")
    except Exception as e:
        print(f"Error during {op_name_or} operation: {e}")
    # Context ctx_or goes out of scope

# -- Example 3: AND Operation on Vector A and Vector B --
    print("\n--- Performing AND operation on Vector A and B ---")
    op_name_and = "AND"
    try:
        ctx_and = scallopy.ScallopContext(provenance="unit")
        ctx_and.add_program(scallop_program_str)
        ctx_and.add_facts("vector_element", facts_a)
        ctx_and.add_facts("vector_element", facts_b)
        request_and_fact = [(id_a, id_b, id_and_ab)]
        ctx_and.add_facts("requested_and_op", request_and_fact)
        print(f"[{op_name_and}] Added request: requested_and_op({id_a}, {id_b}, {id_and_ab})")

        start_time = time.time()
        ctx_and.run()
        end_time = time.time()
        print(f"[{op_name_and}] Scallop run executed in {end_time - start_time:.4f} seconds.")

        scallop_and_result_tuples = list(ctx_and.relation("vector_and_result"))
        print(f"[{op_name_and}] Retrieved {len(scallop_and_result_tuples)} raw result tuples.")

        if scallop_and_result_tuples:
            bool_vec_and_ab = reconstruct_boolean_vector(scallop_and_result_tuples, op_name=op_name_and)
            visualize_boolean_vector(bool_vec_and_ab, title=f"Result: A AND B (Vector {id_and_ab})")
        else:
            print(f"[{op_name_and}] No result found for AND operation.")
    except Exception as e:
        print(f"Error during {op_name_and} operation: {e}")
    # Context ctx_and goes out of scope

    # -- Example 4: XOR Operation on Vector A and Vector B --
    print("\n--- Performing XOR operation on Vector A and B ---")
    op_name_xor = "XOR"
    try:
        ctx_xor = scallopy.ScallopContext(provenance="unit")
        ctx_xor.add_program(scallop_program_str)
        ctx_xor.add_facts("vector_element", facts_a)
        ctx_xor.add_facts("vector_element", facts_b)
        request_xor_fact = [(id_a, id_b, id_xor_ab)]
        ctx_xor.add_facts("requested_xor_op", request_xor_fact)
        print(f"[{op_name_xor}] Added request: requested_xor_op({id_a}, {id_b}, {id_xor_ab})")

        start_time = time.time()
        ctx_xor.run()
        end_time = time.time()
        print(f"[{op_name_xor}] Scallop run executed in {end_time - start_time:.4f} seconds.")

        scallop_xor_result_tuples = list(ctx_xor.relation("vector_xor_result"))
        print(f"[{op_name_xor}] Retrieved {len(scallop_xor_result_tuples)} raw result tuples.")

        if scallop_xor_result_tuples:
            bool_vec_xor_ab = reconstruct_boolean_vector(scallop_xor_result_tuples, op_name=op_name_xor)
            visualize_boolean_vector(bool_vec_xor_ab, title=f"Result: A XOR B (Vector {id_xor_ab})")
        else:
            print(f"[{op_name_xor}] No result found for XOR operation.")
    except Exception as e:
        print(f"Error during {op_name_xor} operation: {e}")
    # Context ctx_xor goes out of scope
