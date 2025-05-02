import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from foolbox import PyTorchModel, attacks, criteria
from PIL import Image

# Settings for toggles and intensity
features = {
    'jpeg_compression': True,  # Toggle for JPEG compression
    'geometric_warp': False,    # Toggle for geometric warp
    'noise_overlay': True,     # Toggle for noise overlay
    'blur': True,              # Toggle for blur
    'sharpen': True,           # Toggle for sharpen
    'pixel_shuffle': False,     # Toggle for pixel shuffle
    'channel_shuffle': True,   # Toggle for channel shuffle
    'color_jitter': True,      # Toggle for color jitter
    'median_blur': True,       # Toggle for median blur
    'universal_patch': True,   # Toggle for universal patch
    'feature_attack': True,    # Toggle for feature attack
    'save_image': True,        # Toggle for saving final image
}

intensities = {
    'jpeg_compression': 0.5,  # Intensity of JPEG compression (0 to 1)
    'geometric_warp': 0.3,    # Intensity of geometric warp (0 to 1)
    'noise_overlay': 0.1,     # Intensity of noise overlay (0 to 1)
    'blur': 0.2,              # Intensity of blur (0 to 1)
    'sharpen': 0.3,           # Intensity of sharpen (0 to 1)
    'pixel_shuffle': 0.5,     # Intensity of pixel shuffle (0 to 1)
    'channel_shuffle': 0.4,   # Intensity of channel shuffle (0 to 1)
    'color_jitter': 0.3,      # Intensity of color jitter (0 to 1)
    'median_blur': 0.2,       # Intensity of median blur (0 to 1)
    'universal_patch': 0.03,  # Epsilon for universal patch (attack intensity)
    'feature_attack': 0.03,   # Epsilon for feature attack (attack intensity)
}

# Load input image
input_path = 'input.jpg'
output_path = 'output.jpg'
img = cv2.imread(input_path)

if img is None:
    print("❌ ERROR: Input image 'input.jpg' not found. Please check the file path.")
    input("Press any key to continue . . .")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")
model = models.resnet18(weights='DEFAULT').eval().to(device)
fmodel = PyTorchModel(model, bounds=(0, 1))

# Apply features based on toggles and intensities
if features['jpeg_compression']:
    _, img_jpeg = cv2.imencode('.jpg', (img_rgb * 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), int(50 * intensities['jpeg_compression'])])
    img_rgb = cv2.imdecode(img_jpeg, cv2.IMREAD_COLOR) / 255.0
    print("✅ JPEG recompression applied")

if features['geometric_warp']:
    rows, cols, _ = img.shape
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    dst_points = np.float32([[0, 0], [cols - 1, 0], [int(0.33 * cols), rows - 1]])
    matrix = cv2.getAffineTransform(src_points, dst_points)
    img_rgb = cv2.warpAffine((img_rgb * 255).astype(np.uint8), matrix, (cols, rows)) / 255.0
    print("✅ Geometric warp applied")

if features['noise_overlay']:
    noise = np.random.normal(0, intensities['noise_overlay'], img_rgb.shape)
    img_rgb = np.clip(img_rgb + noise, 0, 1)
    print("✅ Noise overlay applied")

if features['blur']:
    img_rgb = cv2.GaussianBlur((img_rgb * 255).astype(np.uint8), (5, 5), 0) / 255.0
    print("✅ Blur applied")

if features['sharpen']:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_rgb = cv2.filter2D((img_rgb * 255).astype(np.uint8), -1, kernel) / 255.0
    print("✅ Sharpen applied")

if features['pixel_shuffle']:
    flat_img = img_rgb.reshape(-1, 3)
    np.random.shuffle(flat_img)
    img_rgb = flat_img.reshape(img_rgb.shape)
    print("✅ Pixel shuffle applied")

if features['channel_shuffle']:
    img_rgb = img_rgb[..., np.random.permutation(3)]
    print("✅ Channel shuffle applied")

if features['color_jitter']:
    img_rgb = np.clip(img_rgb * np.random.uniform(1 - intensities['color_jitter'], 1 + intensities['color_jitter']), 0, 1)
    print("✅ Color jitter applied")

if features['median_blur']:
    img_rgb = cv2.medianBlur((img_rgb * 255).astype(np.uint8), 5) / 255.0
    print("✅ Median blur applied")

# Update tensor after all above transforms
img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)

# Foolbox attacks with fail-safe
epsilons = intensities['universal_patch']

# Universal patch (FGSM)
if features['universal_patch']:
    try:
        # Define labels for the Misclassification criterion
        target_class = torch.randint(0, 1000, (1,)).to(device)  # Example: random target class
        criterion = criteria.Misclassification(target_class)
        attack = attacks.LinfFastGradientAttack()
        raw_advs, clipped_advs, success = attack(fmodel, img_tensor, criterion, epsilons=epsilons)
        img_rgb = clipped_advs.squeeze(0).permute(1, 2, 0).cpu().numpy()
        print("✅ Universal patch applied")
    except Exception as e:
        print(f"⚠️ Universal patch failed: {e}")

# Feature attack (BIM)
if features['feature_attack']:
    try:
        # Define the criterion for the attack
        criterion = criteria.Misclassification(target_class)
        attack = attacks.LinfBasicIterativeAttack()
        raw_advs, clipped_advs, success = attack(fmodel, img_tensor, criterion, epsilons=epsilons)
        img_rgb = clipped_advs.squeeze(0).permute(1, 2, 0).cpu().numpy()
        print("✅ Feature attack applied")
    except Exception as e:
        print(f"⚠️ Feature attack failed: {e}")

# Save the final image
if features['save_image']:
    try:
        img_pil = transforms.ToPILImage()(torch.from_numpy(np.clip(img_rgb, 0, 1).transpose(2, 0, 1)))
        img_pil.save(output_path)
        print(f"✅ Final image saved to {output_path}")
    except Exception as e:
        print(f"❌ Failed to save image: {e}")

input("Press any key to continue . . .")
