import matplotlib.pyplot as plt
from skimage import io, img_as_float
import bm3d
import cv2


# # Path to the testing folder
# path = "E:/Bordo/AdvancedImageProcessing/Advimgdata/testing"

# # Get all image files in the directory
# image_files = [f for f in os.listdir(path) if f.endswith('.png') or f.endswith('.j[g]')]

path = "E:/Bordo/AdvancedImageProcessing/Advimgdata/test/00063446_i.png"


# Load the image as a float
noisy_img = img_as_float(io.imread(path, as_gray=True))

# Apply BM3D denoising
BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

# Plot both the original and denoised images side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))


# Display the noisy image
ax[0].imshow(noisy_img, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

# Display the denoised image
ax[1].imshow(BM3D_denoised_image, cmap='gray')
ax[1].set_title("Denoised Image (BM3D)")
ax[1].axis('off')

# Show the plot
plt.show()


#below to run for all Dataset make the output plot correct

# import matplotlib.pyplot as plt
# from skimage import io, img_as_float
# import bm3d
# import os

# # Path to the testing folder
# path = "E:/Bordo/AdvancedImageProcessing/Advimgdata/testing"

# # Get all image files in the directory
# image_files = [f for f in os.listdir(path) if f.endswith('.png') or f.endswith('.png')]

# # Create a figure to display the images
# fig, axes = plt.subplots(len(image_files), 2, figsize=(12, len(image_files)*6))

# for idx, image_file in enumerate(image_files):
#     # Load the image as a float
#     noisy_img = img_as_float(io.imread(os.path.join(path, image_file), as_gray=True))
    
#     # Apply BM3D denoising
#     BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.5, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    
#     # Display the noisy image
#     axes[idx, 0].imshow(noisy_img, cmap='gray')
#     axes[idx, 0].set_title(f"Original Image ({image_file})")
#     axes[idx, 0].axis('off')
    
#     # Display the denoised image
#     axes[idx, 1].imshow(BM3D_denoised_image, cmap='gray')
#     axes[idx, 1].set_title(f"Denoised Image ({image_file})")
#     axes[idx, 1].axis('off')

# # Show the plot
# plt.tight_layout()
# plt.show()

