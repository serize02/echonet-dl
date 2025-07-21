import os
import cv2
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from scipy.spatial.distance import pdist

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_meta(split=None):
    if not os.path.exists('data/dynamic-annot/FileList.csv'):
        raise FileNotFoundError("Metadata file not found at 'data/annotations/metadata.csv'")

    meta = pd.read_csv('data/dynamic-annot/FileList.csv')

    if 'Split' not in meta.columns:
        raise ValueError("'Split' column not found in metadata.csv")

    if split is not None:
        if split not in meta['Split'].unique():
            raise ValueError(f"Split '{split}' not found in metadata")
        meta = meta[meta['Split'] == split]

    return meta


def estimate_L(mask: np.ndarray) -> float:
    if not isinstance(mask, np.ndarray):
        raise TypeError("Mask must be a NumPy array")

    if mask.size == 0:
        return 0.0

    if np.sum(mask) == 0:
        return 0.0

    coords = np.column_stack(np.where(mask == 1))

    if coords.shape[0] < 2:
        return 0.0 

    distances = pdist(coords, metric='euclidean')

    if distances.size == 0 or not np.all(np.isfinite(distances)):
        return 0.0

    return float(np.max(distances))


def fbf_prediction(model, device, video_path, pixel_spacing=0.1, max_frames=100):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    masks, areas, lengths = [], [], []
    frames = 0

    with torch.no_grad():
        while cap.isOpened() and frames < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print(f"Error converting frame to grayscale at frame {frames}: {e}")
                continue

            tensor = image_transforms(gray).unsqueeze(0).float().to(device)

            try:
                pred = model(tensor)
                if pred is None or not torch.is_tensor(pred):
                    raise ValueError("Model prediction is not a valid tensor")

                curr_mask = (pred > 0.5).float().cpu().numpy().squeeze().astype(np.uint8)

                if curr_mask.ndim != 2:
                    raise ValueError(f"Invalid mask shape: {curr_mask.shape}")

                masks.append(curr_mask)
                areas.append(np.sum(curr_mask))
                lengths.append(estimate_L(curr_mask))

            except Exception as e:
                print(f"Error during prediction at frame {frames}: {e}")
                continue

            frames += 1

    cap.release()

    if len(areas) == 0 or len(lengths) == 0:
        raise ValueError("No valid frames were processed.")

    return masks, areas, lengths


def estimate_ef(model, device, video_path, pixel_spacing=0.1):
    masks, areas, lengths = fbf_prediction(model, device, video_path, pixel_spacing=pixel_spacing)

    dice_overlap = []

    prev = None
    for curr in masks:
        if prev is not None:
            try:
                denominator = prev.sum() + curr.sum()
                if denominator == 0:
                    dice = 0.0
                else:
                    intersection = np.logical_and(prev, curr).sum()
                    dice = (2.0 * intersection) / (denominator + 1e-5)
                dice_overlap.append(dice)
            except Exception as e:
                print(f"Dice computation error: {e}")
        prev = curr

    volumes = []

    for area_px, length in zip(areas, lengths):
        try:
            length_mm = length * pixel_spacing
            area_mm2 = area_px * (pixel_spacing ** 2)
            volume = area_mm2 * length_mm

            if not np.isfinite(volume):
                continue

            volumes.append(volume)

        except Exception as e:
            print(f"Volume computation error: {e}")
            continue

    volumes = np.array(volumes)

    if volumes.size == 0:
        raise ValueError("No valid volumes computed.")

    edv = np.max(volumes)
    esv = np.min(volumes)

    ef = ((edv - esv) / edv) * 100 if edv > 0 else 0.0

    length_min = np.min(lengths)
    length_max = np.max(lengths)

    results = {
        'predicted_ef': float(ef),
        'volume_ratio': float(esv / edv) if edv > 0 else 0.0,
        'length_ratio': float(length_min / length_max) if length_max > 0 else 0.0,
        'dice_overlap_std': float(np.std(dice_overlap)),
        'dice_overlap_ratio': float(np.min(dice_overlap) / np.max(dice_overlap))
    }

    return results