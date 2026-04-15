import cv2
import numpy as np
import random


def train_transform(image, mask, size=(512, 512)):
    # Resize first
    image = cv2.resize(image, size)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    # -------------------------
    # Augmentations
    # -------------------------

    # Horizontal flip
    if random.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)

    # Vertical flip
    if random.random() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)

    # Rotation (small)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)

        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

        image = cv2.warpAffine(image, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    # Brightness / contrast
    if random.random() > 0.5:
        alpha = random.uniform(0.8, 1.2)  # contrast
        beta = random.uniform(-20, 20)    # brightness
        image = np.clip(alpha * image + beta, 0, 255)

    # Normalize
    image = image / 255.0

    return image.astype(np.float32), mask.astype(np.float32)

def val_transform(image, mask, size=(512, 512)):
    image = cv2.resize(image, size)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    image = image / 255.0

    return image.astype(np.float32), mask.astype(np.float32)