Semantic-Segmentationin Diverse Weather for Autonomous Driving

##  Introduction

This repository focuses on **semantic segmentation** of street scenes under two different weather conditions: **sunny** and **rainy**. The objective is to segment key elements of the street scene (roads, vehicles, pedestrians, static objects, etc.) using a **U-Net model trained from scratch**—without relying on pre-trained weights.

Rainy conditions introduce significant noise (e.g., raindrops, blurred objects), which affects segmentation accuracy. To address this, the project applies **data preprocessing and denoising techniques** to enhance mask quality, allowing the model to learn effectively despite environmental challenges.

---

## Dataset

The dataset consists of **video frames** and their corresponding **segmentation masks**:

| Condition | Frames | Masks |
|-----------|--------|--------|
| Sunny     | 3779   | 3779   |
| Rainy     | 3642   | 3642   |

![Image](https://github.com/user-attachments/assets/4612be83-f114-405b-9e3a-55a0859fa973)

### Dataset Structure:
Each mask is a **color-coded segmentation map** with **34 classes** (e.g., Road, Sidewalk, Person, Car). Masks are converted from RGB to class IDs (0–33) during preprocessing for training.

Example classes:

| ID  | Class Name       | Color (RGB)        |
|-----|------------------|--------------------|
| 0   | Unlabeled        | (0, 0, 0)          |
| 7   | Road             | (128, 64, 128)     |
| 24  | Person           | (220, 20, 60)      |
| 26  | Car              | (0, 0, 142)        |
| ... | ...              | ...                |
| 33  | Bicycle          | (119, 11, 32)      |


 **Denoising:**
   - Denoising is applied to reduce noise (e.g., raindrops) in the masks, improving segmentation quality—especially for rainy images.
   - A **median filter** is used to smooth out small-scale noise while preserving main structures (roads, vehicles, etc.).
   - **Alternative methods tested:**
     - **Deep Image Prior (DIP):** Neural network-based denoiser.
     - **BM3D:** Block-matching 3D filter for Gaussian noise.  

## 📦 Requirements

The following packages are required:

| Package       | Version (recommended) |
|---------------|-----------------------|
| Python        | 3.7+                  |
| TensorFlow    | 2.x                   |
| Keras         | Included in TF        |
| NumPy         | >=1.19                |
| OpenCV        | >=4.2                 |
| Matplotlib    | >=3.2                 |
| scikit-image  | (for optional BM3D)   |
