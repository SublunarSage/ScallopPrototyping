import cv2
import numpy as np
import matplotlib.pyplot as plt
import scallopy
import os
import time
from rapidocr_onnxruntime import RapidOCR

# --- Configuration ---
SCALLOP_PROGRAM_PATH = "./scallop_bitwise/bitwise.scl"
INPUT_IMAGE_PATH = "./scallop_bitwise/handwritten_input6.png"
CONTOUR_MIN_AREA = 50 # Filter out tiny noise contours
DEBUG_VISUALIZATION = True # Show intermediate steps

# Vector IDs for Scallop
ID_A = 0
ID_B = 1
ID_RESULT = 10 # Generic ID for output


# Define the set of valid characters we expect from OCR
VALID_OCR_CHARS = set(['0', '1', 'A', 'N', 'D', 'O', 'R', 'X'])

# --- Initialize RapidOCR Engine ---
# This will download default models on first run if specific paths are not given.
# The default models typically include robust English recognition.
# Using just RapidOCR() uses default models for detection, classification (optional), and recognition.
# For recognizing pre-cropped characters, detection is less critical but harmless.
print("Initializing RapidOCR engine...")
try:
    ocr_engine = RapidOCR(
        params={"Global.lang_det": "en_mobile", "Global.lang_rec": "en_mobile",},
        use_det=False, use_cls=True, use_rec=True
        
    )
    print("RapidOCR engine initialized successfully.")
except Exception as e:
    print(f"Error initializing RapidOCR engine: {e}")
    print("Make sure you have run 'pip install rapidocr-onnxruntime onnxruntime'")
    exit()

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


    # Sort bounding boxes by x-coordinate (left-to-right)
    bounding_boxes.sort(key=lambda box: box[0])

    if debug_viz and img_for_viz is not None:
        plt.figure(figsize=(max(8, len(bounding_boxes)*0.5),3) ) # Adjust for the number of boxes... 
        plt.imshow(cv2.cvtColor(img_for_viz, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Contours ({len(bounding_boxes)} sorted)")
        plt.show()


    return img, thresh, bounding_boxes #  Pass 'img' (original grayscale) to OCR for patches

def extract_patches_and_recognize_ocr(original_image, bounding_boxes, debug_viz=False):
    """
    Extracts character patches based on bounding_boxes from the original_image
    and uses RapidOCR to recognize them.
    """
    recognized_chars = []

    if ocr_engine is None:
        print("Error: OCR engine not initialized!")
        return []

    patches_for_viz = [] if debug_viz else None

    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Extract the character patch from the *original grayscale image*
        # Add a small padding to the patch, OCR might perform better
        padding = 5 # Pixels
        y_start = max(0, y - padding)
        y_end = min(original_image.shape[0], y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(original_image.shape[1], x + w + padding)
        patch = original_image[y_start:y_end, x_start:x_end]

        if patch.size == 0:
            print(f"Warning: Empty patch for box {i} at ({x},{y},{w},{h}). Skipping.")
            continue

        if debug_viz and patches_for_viz is not None:
            patches_for_viz.append(patch)
        
        # Perform OCR on the patch
        # RapidOCR expects a BGR image or Grayscale. Grayscale is fine.
        ocr_output, elapse = ocr_engine(patch) # ocr_output is [['text', score]], elapse is time

        ocr_text = ""
        confidence = 0.0

        if ocr_output and isinstance(ocr_output, list) and len(ocr_output) > 0:
            # When detection is off, RapidOCR often returns a list containing
            # the recognized text and confidence for the given patch.
            # The patch is given in the form of a list like: [['TEXT', score]]
            recognition_item = ocr_output[0]

            if isinstance(recognition_item, list) and len(recognition_item) == 2:
                ocr_text = str(recognition_item[0]).strip().upper()
                confidence = float(recognition_item[1])
            elif isinstance(recognition_item, tuple):
                # Fallback for other possible tuple formats (e.g., ('TEXT', score) or (None, 'TEXT', score))
                if len(recognition_item) == 2: # ('TEXT', score)
                    ocr_text = str(recognition_item[0]).strip().upper()
                    confidence = float(recognition_item[1])
                elif len(recognition_item) == 3 and recognition_item[0] is None: # (None, 'TEXT', score)
                    ocr_text = str(recognition_item[1]).strip().upper() # Text is the second element
                    confidence = float(recognition_item[2]) # Confidence is the third
                else:
                    print(f"Warning: OCR patch {i} had an unhandled tuple structure in recognition_item: {recognition_item}. Full OCR output: {ocr_output}")
            else:
                print(f"Warning: OCR patch {i} had an unexpected structure for recognition_item: {recognition_item}. Full OCR output: {ocr_output}")

        # If ocr_output is None or an empty list, ocr_text remains "" and confidence 0.0.
        # The 'else' block for "if ocr_text:" will catch this.

        if ocr_text:
            # Take the first character if OCR returns multiple (e.g., "1." -> "1")
            char_candidate = ocr_text[0]
            if char_candidate in VALID_OCR_CHARS:
                recognized_chars.append(char_candidate)
                print(f"OCR: Patch {i} -> '{char_candidate}' (from '{ocr_text}') with confidence: {confidence:.2f}")
            else:
                print(f"Warning: OCR for patch {i} -> '{char_candidate}' (from '{ocr_text}') is NOT in VALID_OCR_CHARS. Confidence: {confidence:.2f}. Full OCR Output: {ocr_output}")
        else:
            # This warning will now trigger if ocr_output was empty/None OR if parsing the structure failed.
            print(f"Warning: OCR for patch {i} resulted in empty text after parsing. Raw OCR Output: {ocr_output}")

    if debug_viz and patches_for_viz:
        num_patches = len(patches_for_viz)
        cols = 5
        rows = (num_patches + cols - 1) // cols

        if num_patches > 0:
            plt.figure(figsize=(cols * 2, rows * 2))
            for idx, p_img in enumerate(patches_for_viz):
                plt.subplot(rows, cols, idx + 1)
                plt.imshow(p_img, cmap='gray')
                plt.title(f"P{idx}")
                plt.axis('off')
            plt.suptitle("Patches Sent to OCR")
            plt.tight_layout()
            plt.show()

    return recognized_chars



# --- Parsing Logic ---

def parse_recognized_sequence(chars):
    """Parses the list of recognized chars into operands and operator."""
    operand1_str = ""
    operand2_str = ""
    operator_str = ""
    
    # States: PARSING_OP1, PARSING_OPERATOR_CHARS, PARSING_OP2
    current_state = "PARSING_OP1"

    temp_operator_chars = [] # To build operator character by character

    for char_idx, char_val in enumerate(chars):
        char = str(char_val).upper() # Ensure it's an uppercase string.

        if current_state == "PARSING_OP1":
            if char in ('0', '1'):
                operand1_str += char
            elif char in ('A','O','X'): # Potential start of an operator
                temp_operator_chars.append(char)
                current_state = "PARSING_OPERATOR_CHARS"
            else:
                raise ValueError(f"Error (Operand 1): Unexpected char '{char}' at index {char_idx}. Operand1: '{operand1_str}'")
        
        elif current_state == "PARSING_OPERATOR_CHARS":
            current_built_op = "".join(temp_operator_chars)
            potential_op = current_built_op + char

            # Check if current character extends a known operator
            if char in ('0', '1'): # Digit encountered, operator must be complete
                if current_built_op in ("AND", "OR", "XOR"):
                    operator_str = current_built_op
                    operand2_str += char # This digit is start of operand 2
                    current_state = "PARSING_OP2"
                    temp_operator_chars = [] # Clear temp
                else:
                    raise ValueError(f"Error (Operator): Digit '{char}' appeared after incomplete operator '{current_built_op}' at index {char_idx}.")
            
            # Check for multi-char operator continuations
            elif potential_op == "AN" or \
                 potential_op == "XO": # Intermediate parts of AND, XOR
                temp_operator_chars.append(char)
            elif potential_op == "AND" or \
                 potential_op == "OR" or \
                 potential_op == "XOR":
                # Full operator formed by adding current char.
                # If OR, it could also be "O" then "R".
                if current_built_op == "O" and char == "R": # OR
                     temp_operator_chars.append(char)
                     # operator_str = "".join(temp_operator_chars) # Set below
                     # current_state = "PARSING_OP2" # Set below
                elif potential_op in ("AND", "XOR"):
                    temp_operator_chars.append(char)
                else: # Should not happen if char is not a digit and potential_op is not a valid continuation
                    raise ValueError(f"Error (Operator): Unexpected char '{char}' when building operator from '{current_built_op}' at index {char_idx}.")
                
                # After appending, check if a full operator is formed
                formed_op_check = "".join(temp_operator_chars)
                if formed_op_check in ("AND", "OR", "XOR"):
                    operator_str = formed_op_check
                    current_state = "PARSING_OP2" # Expect operand 2 or end of sequence next
                    temp_operator_chars = []
                elif len(formed_op_check) >=3: # Max operator length
                    raise ValueError(f"Error (Operator): Invalid operator sequence '{formed_op_check}' at index {char_idx}.")
            else: # Character does not extend a known operator and is not a digit
                raise ValueError(f"Error (Operator): Unexpected char '{char}' while building operator from '{current_built_op}' at index {char_idx}.")
            
        elif current_state == "PARSING_OP2":
            if char in ('0', '1'):
                operand2_str += char
            else:
                raise ValueError(f"Error (Operand 2): Unexpected char '{char}' at index {char_idx}. Expected 0 or 1. Operand2: '{operand2_str}'")
        
        else: # Should not happen
            raise ValueError(f"Unknown parsing state: {current_state}")
    
    # After loop, final validation
    if not operand1_str:
        raise ValueError("Parsing failed: Operand 1 is missing.")
    
    # If loop ended while still parsing operator characters, check if it's a valid one
    if temp_operator_chars:
        final_temp_op = "".join(temp_operator_chars)
        if final_temp_op in ("AND", "OR", "XOR") and not operator_str:
            operator_str = final_temp_op
        else: # Incomplete operator at end of sequence
            if not operator_str : # Only an error if no operator was formed before this
                raise ValueError(f"Parsing failed: Sequence ended with incomplete operator '{final_temp_op}'.")

    if not operator_str:
        raise ValueError(f"Parsing failed: Operator is missing.")
    if operator_str not in ("AND", "OR", "XOR"): # Double check recognized operator
        raise ValueError(f"Parsing failed: Invalid final operator '{operator_str}'.")
    if not operand2_str:
        # This could happen if the input is "1 AND" (which is invalid for the problem)
        # The current_state would be PARSING_OP2 but operand2_str is empty
        raise ValueError("Parsing failed: Operand 2 is missing.")
    
    return operand1_str, operator_str, operand2_str



    #return operand1_str, operator_str, operand2_str

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
        # Ensure 'handwritten_input.png' exists and contains clearly separated characters
        # like "1 0 1 A N D 0 1 1" (black characters on white background).
        original_img, img_thresh_for_segmentation, bounding_boxes = preprocess_and_segment(
            INPUT_IMAGE_PATH,
            CONTOUR_MIN_AREA,
            DEBUG_VISUALIZATION
        )
        print(f"Found {len(bounding_boxes)} potential character contours.")

    except FileNotFoundError:
        print(f"Error: Input image '{INPUT_IMAGE_PATH}' not found.")
        print("Please create this image containing a handwritten sequence like '101 AND 011'.")
        exit()
    except Exception as e:
        print(f"Error during image processing: {e}")
        exit()

    # 3. Character Recognition (using RapidOCR)
    print("\nPerforming character recognition with RapidOCR...")
    try:
        # We pass the original_img (grayscale) for OCR, not the thresholded one used for segmentation
        recognized_chars = extract_patches_and_recognize_ocr(original_img, bounding_boxes, debug_viz=DEBUG_VISUALIZATION)
        print(f"OCR Recognized sequence: {' '.join(recognized_chars)}")
        if not recognized_chars:
            print("OCR did not recognize any valid characters. Check contour segmentation and OCR patch results.")
            exit()
    except Exception as e:
        print(f"Error during OCR: {e}")
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