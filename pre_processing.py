import os
import random
import cv2
from tqdm import tqdm
from scipy.ndimage import median_filter
import shutil
import numpy as np

# Tabla de colores para máscaras
color_to_class = {
    (0, 0, 0): 0,          # Unlabeled
    (111, 74, 0): 5,       # Dynamic
    (81, 0, 81): 6,        # Ground
    (128, 64, 128): 7,     # Road
    (244, 35, 232): 8,     # Sidewalk
    (250, 170, 160): 9,    # Parking
    (230, 150, 140): 10,   # Rail track
    (70, 70, 70): 11,      # Building
    (102, 102, 156): 12,   # Wall
    (190, 153, 153): 13,   # Fence
    (180, 165, 180): 14,   # Guard rail
    (150, 100, 100): 15,   # Bridge
    (150, 120, 90): 16,    # Tunnel
    (153, 153, 153): 17,   # Pole
    (250, 170, 30): 19,    # Traffic light
    (220, 220, 0): 20,     # Traffic sign
    (107, 142, 35): 21,    # Vegetation
    (152, 251, 152): 22,   # Terrain
    (70, 130, 180): 23,    # Sky
    (220, 20, 60): 24,     # Person
    (255, 0, 0): 25,       # Rider
    (0, 0, 142): 26,       # Car
    (0, 0, 70): 27,        # Truck
    (0, 60, 100): 28,      # Bus
    (0, 0, 90): 29,        # Caravan
    (0, 0, 110): 30,       # Trailer
    (0, 80, 100): 31,      # Train
    (0, 0, 230): 32,       # Motorcycle
    (119, 11, 32): 33      # Bicycle
}

def preprocess_masks(segmentation_path, output_path):
    """
    Preprocess masks by converting RGB to grayscale.

    Args:
        segmentation_path (str): Path to folder containing segmented images.
        output_path (str): Path to folder where processed masks will be saved.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filenames = os.listdir(segmentation_path)

    for filename in tqdm(filenames, desc="Processing masks"):
        # Load segmentation image
        segmentation = cv2.imread(os.path.join(segmentation_path, filename))
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB)

        # Convert to grayscale mask
        mask = np.zeros((segmentation.shape[0], segmentation.shape[1]), dtype=np.uint8)
        for color, class_id in color_to_class.items():
            mask[np.all(segmentation == np.array(color), axis=-1)] = class_id

        # Save the processed mask
        processed_filename = os.path.splitext(filename)[0] + "_processed.png"
        cv2.imwrite(os.path.join(output_path, processed_filename), mask)

def apply_denoising_and_move(input_dir, mask_dir, eval_dir, eval_mask_dir, exclude_ratio=0.2, kernel_size=(3, 3)):
    """
    Apply denoising to frames and move 20% of data to evaluation directory.

    Args:
        input_dir (str): Directory containing input frames.
        mask_dir (str): Directory containing corresponding masks.
        eval_dir (str): Directory to save evaluation frames.
        eval_mask_dir (str): Directory to save evaluation masks.
        exclude_ratio (float): Ratio of data to reserve for evaluation.
        kernel_size (tuple): Size of the kernel for the median filter.
    """
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    if not os.path.exists(eval_mask_dir):
        os.makedirs(eval_mask_dir)

    # List all frames and ensure masks match
    filenames = os.listdir(input_dir)
    filenames = [f for f in filenames if os.path.splitext(f)[0] + "_processed.png" in os.listdir(mask_dir)]
    random.shuffle(filenames)

    # Split data into training (denoising applied) and evaluation sets
    split_idx = int(len(filenames) * (1 - exclude_ratio))
    training_files = filenames[:split_idx]
    evaluation_files = filenames[split_idx:]

    # Apply denoising to training data
    for filename in tqdm(training_files, desc="Applying denoising to training data"):
        frame_path = os.path.join(input_dir, filename)
        frame = cv2.imread(frame_path)

        # Apply median filter
        denoised_frame = cv2.medianBlur(frame, kernel_size[0])

        # Save back the denoised frame to the original directory
        cv2.imwrite(os.path.join(input_dir, filename), denoised_frame)

    # Move evaluation data
    for filename in tqdm(evaluation_files, desc="Moving evaluation data"):
        # Move frame
        shutil.move(os.path.join(input_dir, filename), os.path.join(eval_dir, filename))

        # Move corresponding mask
        mask_filename = os.path.splitext(filename)[0] + "_processed.png"
        shutil.move(os.path.join(mask_dir, mask_filename), os.path.join(eval_mask_dir, mask_filename))

    print(f"Denoising applied to {len(training_files)} frames.")
    print(f"{len(evaluation_files)} frames and masks moved to evaluation folder.")

# Paths
rainy_frames_dir = "D:/bordeaux/advanceM/data/rainy_images"
sunny_frames_dir = "D:/bordeaux/advanceM/data/sunny_images"
rainy_segmentation_dir = "D:/bordeaux/advanceM/data/rainy_sseg"
sunny_segmentation_dir = "D:/bordeaux/advanceM/data/sunny_sseg"
rainy_masks_dir = "D:/bordeaux/advanceM/data/rainy_masks"
sunny_masks_dir = "D:/bordeaux/advanceM/data/sunny_masks"
eval_rainy_dir = "D:/bordeaux/advanceM/data/eval_rainy_images"
eval_sunny_dir = "D:/bordeaux/advanceM/data/eval_sunny_images"
eval_rainy_mask_dir = "D:/bordeaux/advanceM/data/eval_rainy_masks"
eval_sunny_mask_dir = "D:/bordeaux/advanceM/data/eval_sunny_masks"

# Preprocess masks
print("Processing rainy masks...")
preprocess_masks(rainy_segmentation_dir, rainy_masks_dir)
print("Processing sunny masks...")
preprocess_masks(sunny_segmentation_dir, sunny_masks_dir)

# Apply denoising and split data
print("Processing rainy frames...")
apply_denoising_and_move(rainy_frames_dir, rainy_masks_dir, eval_rainy_dir, eval_rainy_mask_dir, exclude_ratio=0.2, kernel_size=(3, 3))
print("Processing sunny frames...")
apply_denoising_and_move(sunny_frames_dir, sunny_masks_dir, eval_sunny_dir, eval_sunny_mask_dir, exclude_ratio=0.2, kernel_size=(3, 3))