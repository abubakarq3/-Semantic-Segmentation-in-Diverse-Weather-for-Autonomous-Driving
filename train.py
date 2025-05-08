import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# U-Net model
def unet(input_shape, num_classes):
    """
    Construcción de la arquitectura U-Net desde cero.

    Args:
        input_shape (tuple): Dimensiones de entrada de la imagen (alto, ancho, canales).
        num_classes (int): Número de clases para la segmentación.

    Returns:
        Model: Modelo U-Net compilado.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = tf.keras.layers.UpSampling2D((2, 2))(c5)
    u6 = tf.keras.layers.Concatenate()([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = tf.keras.layers.UpSampling2D((2, 2))(c6)
    u7 = tf.keras.layers.Concatenate()([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = tf.keras.layers.UpSampling2D((2, 2))(c7)
    u8 = tf.keras.layers.Concatenate()([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = tf.keras.layers.UpSampling2D((2, 2))(c8)
    u9 = tf.keras.layers.Concatenate()([u9, c1])
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    # Output layer
    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    # Compile model
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', iou_metric, pixel_accuracy_metric])

    return model

# IoU metric
def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + tf.keras.backend.epsilon())

# Pixel accuracy metric
def pixel_accuracy_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    correct_pixels = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    total_pixels = tf.size(y_true, out_type=tf.int32)  # Solución: Tipo entero válido
    return correct_pixels / tf.cast(total_pixels, tf.float32)

# Data loading
def load_data(image_dir, mask_dir, image_size, fraction=1.0):
    images, masks = [], []
    image_filenames = sorted(os.listdir(image_dir))[:int(len(os.listdir(image_dir)) * fraction)]
    mask_filenames = sorted(os.listdir(mask_dir))[:int(len(os.listdir(mask_dir)) * fraction)]
    for img_file, mask_file in tqdm(zip(image_filenames, mask_filenames), total=len(image_filenames), desc="Loading data"):
        img = tf.keras.preprocessing.image.load_img(os.path.join(image_dir, img_file), target_size=image_size)
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        images.append(img)
        mask = tf.keras.preprocessing.image.load_img(os.path.join(mask_dir, mask_file), color_mode="grayscale", target_size=image_size)
        mask = tf.keras.preprocessing.image.img_to_array(mask).astype(np.uint8)
        masks.append(mask)
    return np.array(images), np.array(masks)

# Generator
def mask_generator(images, masks, num_classes, batch_size):
    while True:
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_masks = masks[i:i + batch_size]
            batch_masks_categorical = np.array([tf.keras.utils.to_categorical(mask, num_classes=num_classes) for mask in batch_masks])
            yield batch_images, batch_masks_categorical

# Paths
image_size = (128, 128)
num_classes = 34
batch_size = 16
fraction = 1.0
output_dir_metrics = "output_metrics"
if not os.path.exists(output_dir_metrics):
    os.makedirs(output_dir_metrics)

# Load data
sunny_image_dir = "D:/bordeaux/advanceM/data/sunny_images"
sunny_mask_dir = "D:/bordeaux/advanceM/data/sunny_masks"
rainy_image_dir = "D:/bordeaux/advanceM/data/rainy_images"
rainy_mask_dir = "D:/bordeaux/advanceM/data/rainy_masks"

sunny_images, sunny_masks = load_data(sunny_image_dir, sunny_mask_dir, image_size, fraction=fraction)
rainy_images, rainy_masks = load_data(rainy_image_dir, rainy_mask_dir, image_size, fraction=fraction)

all_images = np.concatenate([sunny_images, rainy_images], axis=0)
all_masks = np.concatenate([sunny_masks, rainy_masks], axis=0)

train_images, test_images, train_masks, test_masks = train_test_split(all_images, all_masks, test_size=0.3, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(test_images, test_masks, test_size=0.5, random_state=42)

# Model training
model = unet(input_shape=image_size + (3,), num_classes=num_classes)
checkpoint = tf.keras.callbacks.ModelCheckpoint("model_checkpoint.keras", save_best_only=True, monitor="val_loss", mode="min")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    mask_generator(train_images, train_masks, num_classes, batch_size),
    steps_per_epoch=len(train_images) // batch_size,
    validation_data=mask_generator(val_images, val_masks, num_classes, batch_size),
    validation_steps=len(val_images) // batch_size,
    epochs=15,
    callbacks=[checkpoint, early_stopping]
)

# Plot metrics
plt.figure(figsize=(15, 8))
metrics = ['accuracy', 'loss', 'iou_metric', 'pixel_accuracy_metric']
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    plt.plot(history.history[metric], label=f'Training {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.title(metric.capitalize())
    plt.legend()
    plt.savefig(os.path.join(output_dir_metrics, f"{metric}_plot.png"))
plt.close()