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

    return meta


def estimate_L(mask: np.ndarray) -> float:
    
    if np.sum(mask) == 0:
        return 0.0

    coords = np.column_stack(np.where(mask == 1))
    max_distance = np.max(pdist(coords, metric='euclidean'))

    return float(max_distance)

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

    return areas, lengths
    

def estimate_ef(model, device, video_path, pixel_spacing=0.1):
    
    areas, lengths = fbf_prediction(model, device, video_path)

    volumes = []

    for area_px, length in zip(areas, lengths):
        length_mm = length * pixel_spacing
        area_mm2 = area_px * (pixel_spacing ** 2)
        volume = area_mm2 * length_mm
        volumes.append(volume)

    volumes = np.array(volumes)

    edv = np.max(volumes)
    esv = np.min(volumes)
    ef = ((edv - esv) / edv) * 100 if edv > 0 else 0.0

    results = {
        'predicted_ef': float(ef),
        'volume_ratio': float(esv / edv) if edv > 0 else float(0.0),
        'length_ratio': float(np.min(lengths) / np.max(lengths)) if np.max(lengths) > 0 else float(0.0),
    }

    return results