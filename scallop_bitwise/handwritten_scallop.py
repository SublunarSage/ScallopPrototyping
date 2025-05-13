import cv2
import numpy as np
import matplotlib.pyplot as plt
import scallopy
import os
import time

# --- Configuration ---
SCALLOP_PROGRAM_PATH = "./scallop_bitwise/bitwise.scl"
INPUT_IMAGE_PATH = "./scallop_bitwise/handwritten_input.png" # CREATE THIS IMAGE (e.g., "101 AND 011")
IMG_SIZE_FOR_RECOGNITION = 28 # Standard MNIST/EMNIST size
CONTOUR_MIN_AREA = 50 # Filter out tiny noise contours
DEBUG_VISUALIZATION = True # Show intermediate steps

# Vector IDs for Scallop
ID_A = 0
ID_B = 1
ID_RESULT = 10 # Generic ID for output

# --- Helper Functions (Adapted from minst_bitwise.py) ---

def format_vector_for_scallop(vector_id, bool_vector):
    """Formats a boolean vector into Scallop facts."""
    return [(vector_id, i, val) for i, val in enumerate(bool_vector)]

def reconstruct_boolean_vector(scallop_result_tuples, expected_length, op_name="Operation"):
    """Reconstructs a boolean vector from Scallop query results."""
    if not scallop_result_tuples:
        print(f"Warning [{op_name}]: No tuples found for reconstruction.")
        return []

    # Extract relevant fields (id, index, value) assuming different result relation structures
    processed_tuples = []
    if scallop_result_tuples:
         # Check tuple length based on the known result relations
         # vector_..._result(output_id: vector_id, index: usize, value: bool) -> length 3
         if len(scallop_result_tuples[0]) == 3:
              processed_tuples = [(item[0], item[1], item[2]) for item in scallop_result_tuples] # (id, index, value)
         else:
              print(f"Error [{op_name}]: Unexpected tuple length {len(scallop_result_tuples[0])}. Expected 3.")
              return []

    # Sort by index to ensure correct order
    try:
        # item[1] is the index
        sorted_result = sorted(processed_tuples, key=lambda item: int(item[1]))
    except (ValueError, TypeError, IndexError) as e:
        print(f"Error [{op_name}]: Could not sort result tuples by index. Invalid data? Error: {e}")
        return []

    num_results = len(sorted_result)
    print(f"Debug [{op_name}]: Received {num_results} tuples after retrieval and processing.")

    # Check for duplicates (optional but good practice)
    indices_present = [item[1] for item in sorted_result]
    if len(indices_present) != len(set(indices_present)):
         print(f"Warning [{op_name}]: Duplicate indices found!")
         # Simple deduplication: keep first encountered after sorting
         unique_results = []
         seen_indices = set()
         for item in sorted_result:
            if item[1] not in seen_indices:
                unique_results.append(item)
                seen_indices.add(item[1])
         sorted_result = unique_results
         num_results = len(sorted_result)

    # Check final count against expected length (which should be known *after* parsing)
    if num_results != expected_length:
        print(f"Warning [{op_name}]: Reconstructed vector length mismatch. Got {num_results} unique tuples, expected {expected_length}.")
        # Decide handling: Padding/truncating usually done *before* Scallop or handled in reconstruct logic
    elif expected_length > 0 : # Only print info if we expect something
         print(f"Info [{op_name}]: Successfully received expected number of results ({expected_length}).")


    # Extract boolean values (item[2] is the value)
    bool_vector = [item[2] for item in sorted_result]

     # Ensure final length matches expected length exactly if needed (e.g., for viz)
    if len(bool_vector) < expected_length:
        print(f"Warning [{op_name}]: Padding reconstructed vector (length {len(bool_vector)}) to expected {expected_length}")
        bool_vector.extend([False] * (expected_length - len(bool_vector)))
    elif len(bool_vector) > expected_length:
        print(f"Warning [{op_name}]: Truncating reconstructed vector (length {len(bool_vector)}) to expected {expected_length}")
        bool_vector = bool_vector[:expected_length]

    return bool_vector


def visualize_boolean_vector(bool_vector, title="Boolean Image", img_width=None):
    """Visualizes a boolean vector as a 1D strip or 2D if width provided."""
    if not bool_vector:
        print(f"Error visualizing '{title}': Input vector is empty.")
        return

    img_array = np.array(bool_vector, dtype=float)

    if img_width and len(bool_vector) % img_width == 0:
        img_height = len(bool_vector) // img_width
        img_array = img_array.reshape((img_height, img_width))
        figsize=(4, 4 * img_height / img_width)
    else:
        # Display as a 1D strip
        img_array = img_array.reshape((1, -1))
        figsize=(8, 1) # Adjust as needed
        if img_width:
             print(f"Warning visualizing '{title}': Cannot reshape {len(bool_vector)} elements into width {img_width}. Displaying as 1D.")


    plt.figure(figsize=figsize)
    plt.imshow(img_array, cmap='gray_r', vmin=0, vmax=1, aspect='auto') # white=False(0), black=True(1)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# --- Image Processing and Simulated Recognition ---

def preprocess_and_segment(image_path, min_area=50, debug_viz=False):
    """Loads, preprocesses, finds contours, and sorts bounding boxes."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Thresholding (adjust threshold value if needed)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV) # Invert if chars are white on black

    if debug_viz:
        plt.figure(figsize=(6, 2))
        plt.imshow(thresh, cmap='gray')
        plt.title("Thresholded Image")
        plt.show()


    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and get bounding boxes
    bounding_boxes = []
    img_for_viz = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if debug_viz else None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
            if debug_viz:
                 cv2.rectangle(img_for_viz, (x, y), (x + w, y + h), (0, 255, 0), 1)


    # Sort bounding boxes by x-coordinate
    bounding_boxes.sort(key=lambda box: box[0])

    if debug_viz and img_for_viz is not None:
        plt.figure(figsize=(8, 3))
        plt.imshow(cv2.cvtColor(img_for_viz, cv2.COLOR_BGR2RGB))
        plt.title("Detected Contours (Sorted Left-to-Right)")
        plt.show()


    return img, thresh, bounding_boxes

# *** SIMULATED RECOGNITION ***
# In a real system, this function would use a trained NN model.
# For this example, we rely on the known order of characters in the image.
# You MUST create your input image (e.g., handwritten_input.png) to match this expected sequence.
EXPECTED_SEQUENCE = ['1', '0', '1', 'A', 'N', 'D', '0', '1', '1'] # Example for "101 AND 011"

def recognize_character_simulated(img_patch, index, expected_sequence):
    """Simulates character recognition based on expected order."""
    if index < len(expected_sequence):
        char = expected_sequence[index]
        print(f"Simulated Recognition: Box {index} -> '{char}'")
        return char
    else:
        print(f"Warning: More bounding boxes ({index+1}) than expected characters ({len(expected_sequence)}).")
        return None # Or raise error

def extract_patches_and_recognize(img_thresh, bounding_boxes, expected_sequence):
    """Extracts patches and uses simulated recognition."""
    recognized_chars = []
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Extract the character patch
        patch = img_thresh[y:y+h, x:x+w]

        # (Optional but good practice) Preprocess patch for NN:
        # - Add padding to make it square
        # - Resize to NN input size (e.g., 28x28)
        # - Normalize pixel values
        # Since recognition is simulated, we skip detailed NN preprocessing.

        # Simulate recognition
        char = recognize_character_simulated(patch, i, expected_sequence)
        if char:
            recognized_chars.append(char)
    return recognized_chars

# --- Parsing Logic ---

def parse_recognized_sequence(chars):
    """Parses the list of recognized chars into operands and operator."""
    operand1_str = ""
    operand2_str = ""
    operator_str = ""
    state = "OPERAND1" # States: OPERAND1, OPERATOR, OPERAND2

    for char in chars:
        if state == "OPERAND1":
            if char in ('0', '1'):
                operand1_str += char
            elif char in ('A', 'O', 'X'): # Start of Operator
                operator_str += char
                state = "OPERATOR"
            else:
                raise ValueError(f"Unexpected character '{char}' while expecting operand 1 or start of operator.")
        elif state == "OPERATOR":
            if char in ('N', 'D', 'R'): # Continuing Operator (AND, OR)
                 operator_str += char
                 # Check for complete operators
                 if operator_str in ("AND", "OR", "XOR"): # XOR is 3 letters
                       if operator_str == "XOR":
                           state = "OPERAND2" # Only move after full XOR
                       # For AND/OR, we might see 'A', then 'N', then 'D'
                       # Or 'O', then 'R'
                       # Need to handle potential next char being a digit
                 elif len(operator_str) > 3:
                     raise ValueError(f"Invalid operator sequence: {operator_str}")

            elif char in ('0', '1'): # Operator finished, start operand 2
                 if operator_str not in ("AND", "OR", "XOR"):
                     # Allow single letter 'X' if followed by digit? Let's assume full words.
                     # For simplicity, require full AND/OR/XOR before digits
                     raise ValueError(f"Incomplete or invalid operator '{operator_str}' before digit '{char}'.")
                 state = "OPERAND2"
                 operand2_str += char
            else:
                 raise ValueError(f"Unexpected character '{char}' while parsing operator '{operator_str}'.")

            # Handle transition after completing AND/OR
            if state != "OPERAND2" and operator_str in ("AND", "OR"):
                 # If the *next* character isn't a digit, it's an error
                 # This check is tricky here, better done after loop or by peeking ahead
                 # Let's simplify: assume operator is fully formed before digits start
                 pass # Handled when digit appears

        elif state == "OPERAND2":
            if char in ('0', '1'):
                operand2_str += char
            else:
                raise ValueError(f"Unexpected character '{char}' while expecting operand 2.")

    # Final validation
    if not operand1_str: raise ValueError("Missing operand 1.")
    if not operator_str: raise ValueError("Missing operator.")
    if operator_str not in ("AND", "OR", "XOR"): raise ValueError(f"Invalid final operator: {operator_str}")
    if not operand2_str: raise ValueError("Missing operand 2.")
    if state != "OPERAND2": raise ValueError("Parsing did not end in the correct state.")


    return operand1_str, operator_str, operand2_str

# --- Main Execution Logic ---

if __name__ == "__main__":
    # 1. Load Scallop Program
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

    # 2. Load Image, Preprocess, Segment
    try:
        print(f"\nProcessing image: {INPUT_IMAGE_PATH}")
        # Create a sample image 'handwritten_input.png' manually with "101 AND 011"
        # Ensure characters are clearly separated. White background, black text is standard.
        # If you draw black on white, the thresholding might need adjustment or inversion.
        # Example: Use MSPaint, GIMP, etc. Font: Handwritten-style, Size: ~48pt
        # Make sure the EXPECTED_SEQUENCE above matches your image content and order.
        img, img_thresh, bounding_boxes = preprocess_and_segment(
            INPUT_IMAGE_PATH,
            CONTOUR_MIN_AREA,
            DEBUG_VISUALIZATION
        )
        print(f"Found {len(bounding_boxes)} potential character contours.")
        if len(bounding_boxes) != len(EXPECTED_SEQUENCE):
             print(f"Warning: Number of contours ({len(bounding_boxes)}) does not match expected sequence length ({len(EXPECTED_SEQUENCE)}). Recognition might be incorrect.")

    except FileNotFoundError:
        print(f"Error: Input image '{INPUT_IMAGE_PATH}' not found.")
        print("Please create this image containing a handwritten sequence like '101 AND 011'.")
        exit()
    except Exception as e:
        print(f"Error during image processing: {e}")
        exit()

    # 3. Simulated Character Recognition
    print("\nSimulating character recognition...")
    try:
        recognized_chars = extract_patches_and_recognize(img_thresh, bounding_boxes, EXPECTED_SEQUENCE)
        print(f"Recognized sequence: {' '.join(recognized_chars)}")
    except Exception as e:
        print(f"Error during simulated recognition: {e}")
        exit()

    # 4. Parse Sequence
    print("\nParsing recognized sequence...")
    try:
        op1_str, op_str, op2_str = parse_recognized_sequence(recognized_chars)
        print(f"Parsed: Operand 1 = {op1_str}, Operator = {op_str}, Operand 2 = {op2_str}")
    except ValueError as e:
        print(f"Error parsing sequence: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during parsing: {e}")
        exit()


    # 5. Convert to Boolean Vectors and Pad
    print("\nConverting operands to boolean vectors...")
    vec_a_bool = [char == '1' for char in op1_str]
    vec_b_bool = [char == '1' for char in op2_str]

    # Pad shorter vector
    len_a = len(vec_a_bool)
    len_b = len(vec_b_bool)
    target_len = max(len_a, len_b)

    if len_a < target_len:
        vec_a_bool.extend([False] * (target_len - len_a))
        print(f"Padded vector A to length {target_len}.")
    if len_b < target_len:
        vec_b_bool.extend([False] * (target_len - len_b))
        print(f"Padded vector B to length {target_len}.")

    print(f"Vector A (ID {ID_A}): {vec_a_bool}")
    print(f"Vector B (ID {ID_B}): {vec_b_bool}")

    # 6. Prepare Scallop Facts
    print("\nPreparing Scallop facts...")
    facts_a = format_vector_for_scallop(ID_A, vec_a_bool)
    facts_b = format_vector_for_scallop(ID_B, vec_b_bool)

    request_facts = []
    result_relation_name = ""
    scallop_op_name = op_str # For logging

    if op_str == "AND":
        request_facts = [(ID_A, ID_B, ID_RESULT)]
        request_relation_name = "requested_and_op"
        result_relation_name = "vector_and_result"
    elif op_str == "OR":
        request_facts = [(ID_A, ID_B, ID_RESULT)]
        request_relation_name = "requested_or_op"
        result_relation_name = "vector_or_result"
    elif op_str == "XOR":
        request_facts = [(ID_A, ID_B, ID_RESULT)]
        request_relation_name = "requested_xor_op"
        result_relation_name = "vector_xor_result"
    else:
        # Should have been caught by parser, but defensive check
        print(f"Error: Unsupported operator '{op_str}' for Scallop operation.")
        exit()

    print(f"Requesting {op_str} operation: {request_relation_name}{request_facts}")

    # 7. Run Scallop
    print(f"\nRunning Scallop for {op_str} operation...")
    try:
        ctx = scallopy.ScallopContext(provenance="unit") # Or another provenance if needed
        ctx.add_program(scallop_program_str)
        ctx.add_facts("vector_element", facts_a)
        ctx.add_facts("vector_element", facts_b)
        ctx.add_facts(request_relation_name, request_facts)

        start_time = time.time()
        ctx.run()
        end_time = time.time()
        print(f"Scallop run executed in {end_time - start_time:.4f} seconds.")

        # 8. Retrieve Result
        scallop_result_tuples = list(ctx.relation(result_relation_name))
        print(f"Retrieved {len(scallop_result_tuples)} raw result tuples from '{result_relation_name}'.")

        # 9. Reconstruct Result Vector
        if scallop_result_tuples:
            result_vec_bool = reconstruct_boolean_vector(scallop_result_tuples, target_len, op_name=scallop_op_name)
            print(f"\n--- Result of {op1_str} {op_str} {op2_str} ---")
            print(f"Boolean Vector (ID {ID_RESULT}): {result_vec_bool}")

            # 10. Visualize Result (Optional)
            visualize_boolean_vector(result_vec_bool, title=f"Result: {op1_str} {op_str} {op2_str}", img_width=target_len) # Visualize as 1xN strip

        else:
            print(f"\n--- No result found for {op_str} operation. ---")
            # Check scallop rules and input facts if this happens unexpectedly.

    except Exception as e:
        print(f"Error during Scallop execution or result processing: {e}")
        # Consider printing more context if debugging provenance issues