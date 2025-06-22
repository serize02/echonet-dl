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
    
    meta = pd.read_csv('data/annotations/metadata.csv')
    
    if split is not None:
        meta = meta[meta['Split'] == split]

    meta = meta[meta.apply(
        lambda x: os.path.exists(os.path.join('data/echonet', x['FileName'] + '.avi')),
        axis=1
    )]

    return meta


def estimate_L(mask: np.ndarray) -> float:
    
    if np.sum(mask) == 0:
        return 0.0

    coords = np.column_stack(np.where(mask == 1))
    max_distance = np.max(pdist(coords, metric='euclidean'))

    return float(max_distance)


def flow_stats(masks, pixel_spacing=0.1):
    divergence_scores = []
    magnitudes = []
    dice_scores = []

    prev = None
    for curr in masks:
        if prev is not None:
            lv_region = curr > 0

            flow = cv2.calcOpticalFlowFarneback(
                prev.astype(np.uint8) * 255,
                curr.astype(np.uint8) * 255,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            vx = flow[..., 0] * pixel_spacing
            vy = flow[..., 1] * pixel_spacing

            mag = np.sqrt(vx**2 + vy**2)
            magnitudes.append(mag[lv_region])

            dx = cv2.Sobel(vx, cv2.CV_64F, 1, 0, ksize=5)
            dy = cv2.Sobel(vy, cv2.CV_64F, 0, 1, ksize=5)
            divergence = dx + dy
            avg_divergence = np.mean(np.abs(divergence[lv_region]))
            divergence_scores.append(avg_divergence)

            intersection = np.logical_and(prev, curr).sum()
            dice = (2.0 * intersection) / (prev.sum() + curr.sum() + 1e-5)
            dice_scores.append(dice)

        prev = curr

    # Flatten all magnitudes
    mags = np.concatenate(magnitudes) if magnitudes else np.array([])

    if mags.size > 0:
        mean_mag = float(np.mean(mags))
        var_mag = float(np.var(mags))
        std_mag = float(np.std(mags))
        max_mag = float(np.max(mags))
    else:
        mean_mag = var_mag = std_mag = max_mag = 0.0

    if len(divergence_scores) > 0:
        mean_div = float(np.mean(divergence_scores))
        var_div = float(np.var(divergence_scores))
        std_div = float(np.std(divergence_scores))
        max_div = float(np.max(divergence_scores))
    else:
        mean_div = var_div = std_div = max_div = 0.0

    if len(dice_scores) > 0:
        mean_dice = float(np.mean(dice_scores))
        var_dice = float(np.var(dice_scores))
        std_dice = float(np.std(dice_scores))
        min_dice = float(np.min(dice_scores))
    else:
        mean_dice = var_dice = std_dice = min_dice = 0.0

    stats = {
        'mean_magnitude': mean_mag,
        'var_magnitude': var_mag,
        'std_magnitude': std_mag,
        'max_magnitude': max_mag,
        'mean_divergence': mean_div,
        'var_divergence': var_div,
        'std_divergence': std_div,
        'max_divergence': max_div,
        'mean_dice': mean_dice,
        'var_dice': var_dice,
        'std_dice': std_dice,
        'min_dice': min_dice,
    }

    return stats



def fbf_prediction(model, device, video_path, pixel_spacing=0.1, max_frames=100):

    cap = cv2.VideoCapture(video_path)

    masks = []
    areas = []
    lengths = []

    frames = 0

    with torch.no_grad():

        while cap.isOpened() and frames < max_frames:

            ret, frame = cap.read()
            if not ret:
              break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tensor = image_transforms(gray).unsqueeze(0).float().to(device)
            pred = model(tensor)
            curr_mask = (pred > 0.5).float().cpu().numpy().squeeze().astype(np.uint8)

            masks.append(curr_mask)
            areas.append(np.sum(curr_mask))
            lengths.append(estimate_L(curr_mask))

            frames += 1

        cap.release()

    return areas, lengths, flow_stats(masks)
    

def estimate_ef(model, device, video_path, pixel_spacing=0.1):
    
    areas, lengths, flow_stats = fbf_prediction(model, device, video_path)

    volumes = []

    for area_px, length in zip(areas, lengths):
        length_cm = length * pixel_spacing
        area_cm2 = area_px * (pixel_spacing ** 2)
        volume = area_cm2 * length_cm
        volumes.append(volume)

    volumes = np.array(volumes)

    edv = np.max(volumes)
    esv = np.min(volumes)
    ef = ((edv - esv) / edv) * 100 if edv > 0 else 0.0

    stats = {
        'predicted_ef': ef,
        'volume_range': edv - esv,
        'volume_mean': np.mean(volumes),
        'volume_std': np.std(volumes),
        'volume_max': np.max(volumes),
        'volume_min': np.min(volumes),
        'volume_ratio': esv / edv if edv > 0 else 0.0,
        'length_mean': np.mean(lengths),
        'length_std': np.std(lengths),
        'length_range': np.max(lengths) - np.min(lengths),
        'area_mean': np.mean(areas),
        'area_std': np.std(areas),
        'area_range': np.max(areas) - np.min(areas),
    }

    return stats | flow_stats