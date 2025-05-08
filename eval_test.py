import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Color mapping table
color_to_class = {
    (0, 0, 0): 0, (111, 74, 0): 5, (81, 0, 81): 6, (128, 64, 128): 7,
    (244, 35, 232): 8, (250, 170, 160): 9, (230, 150, 140): 10, (70, 70, 70): 11,
    (102, 102, 156): 12, (190, 153, 153): 13, (180, 165, 180): 14, (150, 100, 100): 15,
    (150, 120, 90): 16, (153, 153, 153): 17, (250, 170, 30): 19, (220, 220, 0): 20,
    (107, 142, 35): 21, (152, 251, 152): 22, (70, 130, 180): 23, (220, 20, 60): 24,
    (255, 0, 0): 25, (0, 0, 142): 26, (0, 0, 70): 27, (0, 60, 100): 28,
    (0, 0, 90): 29, (0, 0, 110): 30, (0, 80, 100): 31, (0, 0, 230): 32, (119, 11, 32): 33
}
class_to_color = {v: k for k, v in color_to_class.items()}

def remap_to_rgb(mask):
    """
    Converts a grayscale mask to RGB using the color mapping table.
    """
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in class_to_color.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask

def predict_images(image_folder, model, image_size, output_folder):
    """
    Processes images, generates predictions, and saves results.

    Args:
        image_folder (str): Path to the folder containing evaluation images.
        model (tf.keras.Model): Pre-trained U-Net model.
        image_size (tuple): Model's input size (height, width).
        output_folder (str): Directory to save predictions.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))])

    for img_file in tqdm(image_filenames, desc="Processing images"):
        img_path = os.path.join(image_folder, img_file)

        # Load image
        img = tf.keras.preprocessing.image.load_img(img_path)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        original_size = img_array.shape[:2]

        # Resize for model input
        resized_img = tf.image.resize(img_array, image_size)
        resized_img = np.expand_dims(resized_img, axis=0)

        # Predict mask
        pred_mask = model.predict(resized_img)
        pred_mask = np.argmax(pred_mask, axis=-1).squeeze()

        # Expand dimensions to match tf.image.resize requirements
        pred_mask_expanded = np.expand_dims(pred_mask, axis=-1)

        # Resize prediction back to original size
        pred_mask_resized = tf.image.resize(pred_mask_expanded, original_size, method="nearest").numpy().squeeze().astype(np.uint8)

        # Convert prediction to RGB
        pred_rgb = remap_to_rgb(pred_mask_resized)

        # Save comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(pred_rgb)
        axes[1].set_title("Prediction")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_prediction.png"))
        plt.close(fig)

# Parameters
image_size = (128, 128)
eval_rainy_images = "D:/bordeaux/advanceM/data/eval_rainy_images"
eval_sunny_images = "D:/bordeaux/advanceM/data/eval_sunny_images"
output_folder_rainy = "D:/bordeaux/advanceM/output_rainy_predictions"
output_folder_sunny = "D:/bordeaux/advanceM/output_sunny_predictions"

# Load model
model = tf.keras.models.load_model(
    "D:/bordeaux/advanceM/model_checkpoint.keras",
    custom_objects={"iou_metric": lambda y_true, y_pred: y_pred, "pixel_accuracy_metric": lambda y_true, y_pred: y_pred}
)

# Evaluate Rainy images
print("Processing Rainy dataset...")
predict_images(eval_rainy_images, model, image_size, output_folder_rainy)

# Evaluate Sunny images
print("Processing Sunny dataset...")
predict_images(eval_sunny_images, model, image_size, output_folder_sunny)