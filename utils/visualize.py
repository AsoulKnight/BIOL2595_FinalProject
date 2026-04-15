import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch

def load_mask(mask_path):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Load as grayscale
    mask = np.array(Image.open(mask_path).convert("L"))

    # Convert to binary
    mask = (mask > 0).astype(np.uint8)

    return mask


####### Sanity check #######
def visualize_sample(image_path, mask_path):
    # Load image (RGB)
    image = np.array(Image.open(image_path))

    # Load mask
    mask = load_mask(mask_path)

    # Plot
    plt.figure(figsize=(10, 4))

    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    # Mask
    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask (Binary)")
    plt.axis('off')

    # Overlay (red)
    plt.subplot(1, 4, 3)
    plt.imshow(image)
    plt.imshow(mask * 255, cmap='Reds', alpha=0.5)
    plt.title("Overlay")
    plt.axis('off')
    # Overlay (blue )
    plt.subplot(1, 4, 4)
    plt.imshow(image)
    plt.imshow(mask * 255, cmap='jet', alpha=0.5)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# image_path1 = "C:/1.lwBrown/Lee Lab/pythonProject/Models/Datasets/STARE/vessel_segmentation/images/im0001.ppm"
# mask_path1= "C:/1.lwBrown/Lee Lab/pythonProject/Models/Datasets/STARE/vessel_segmentation/masks_ah/im0001.ah.ppm"
# image_path2 = "C:/1.lwBrown/Lee Lab/pythonProject/Models/Datasets/STARE/vessel_segmentation/images/im0005.ppm"
# mask_path2= "C:/1.lwBrown/Lee Lab/pythonProject/Models/Datasets/STARE/vessel_segmentation/masks_ah/im0005.ah.ppm"
# image_path3 = "C:/1.lwBrown/Lee Lab/pythonProject/Models/Datasets/STARE/vessel_segmentation/images/im0077.ppm"
# mask_path3= "C:/1.lwBrown/Lee Lab/pythonProject/Models/Datasets/STARE/vessel_segmentation/masks_ah/im0077.ah.ppm"
# image_path4 = "C:/1.lwBrown/Lee Lab/pythonProject/Models/Datasets/STARE/vessel_segmentation/images/im0139.ppm"
# mask_path4= "C:/1.lwBrown/Lee Lab/pythonProject/Models/Datasets/STARE/vessel_segmentation/masks_ah/im0139.ah.ppm"
# image_path5 = "C:/1.lwBrown/Lee Lab/pythonProject/Models/Datasets/STARE/vessel_segmentation/images/im0319.ppm"
# mask_path5= "C:/1.lwBrown/Lee Lab/pythonProject/Models/Datasets/STARE/vessel_segmentation/masks_ah/im0319.ah.ppm"
#
# # ##### Sanity check visualization ######
# visualize_sample(image_path1, mask_path1)
# visualize_sample(image_path2, mask_path2)
# visualize_sample(image_path3, mask_path3)
# visualize_sample(image_path4, mask_path4)
# visualize_sample(image_path5, mask_path5)





def save_predictions(model, loader, device, save_dir, fold, num_samples=3):
    model.eval()

    count = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            for i in range(images.size(0)):
                if count >= num_samples:
                    return

                img = images[i].cpu().permute(1, 2, 0).numpy()
                gt = masks[i].cpu().squeeze().numpy()
                pred = preds[i].cpu().squeeze().numpy()

                plt.figure(figsize=(10, 3))

                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title("Image")
                plt.axis("off")

                plt.subplot(1, 3, 2)
                plt.imshow(gt, cmap="gray")
                plt.title("Ground Truth")
                plt.axis("off")

                plt.subplot(1, 3, 3)
                plt.imshow(pred, cmap="gray")
                plt.title("Prediction")
                plt.axis("off")

                save_path = f"{save_dir}/fold{fold}_pred_{count}.png"
                plt.savefig(save_path)
                plt.close()

                count += 1