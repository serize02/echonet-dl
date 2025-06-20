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

def navier_stokes_flow_divergence(masks, pixel_spacing=0.1):
    
    divergence_scores = []
    prev = None
    for curr in masks:
        if prev is not None:

            flow = cv2.calcOpticalFlowFarneback(
                prev.astype(np.uint8) * 255,
                curr.astype(np.uint8) * 255,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            vx = flow[..., 0] * pixel_spacing
            vy = flow[..., 1] * pixel_spacing

            dx = cv2.Sobel(vx, cv2.CV_64F, 1, 0, ksize=5)
            dy = cv2.Sobel(vy, cv2.CV_64F, 0, 1, ksize=5)

            divergence = dx + dy

            mask_region = curr > 0
            avg_divergence = np.mean(np.abs(divergence[mask_region]))

            divergence_scores.append(avg_divergence)
        prev = curr

    return np.mean(divergence_scores) if divergence_scores else 0.0


def dice_stability(masks):
    dice_scores = []
    prev = None
    for curr in masks:
        if prev is not None:
            intersection = np.logical_and(prev, curr).sum()
            dice = (2.0 * intersection) / (prev.sum() + curr.sum() + 1e-5)
            dice_scores.append(dice)
        prev = curr
    return np.mean(dice_scores) if dice_scores else 0.0


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

    return areas, lengths, navier_stokes_flow_divergence(masks), dice_stability(masks)
    

def estimate_ef(areas, lengths, pixel_spacing=0.1):

    volumes = []

    for area_px, length in zip(areas, lengths):
        length_cm = length * pixel_spacing
        area_cm2 = area_px * (pixel_spacing ** 2)
        volume = area_cm2 * length_cm
        volumes.append(volume)

    edv = max(volumes)
    esv = min(volumes)
    ef = ((edv - esv) / edv) * 100

    return ef