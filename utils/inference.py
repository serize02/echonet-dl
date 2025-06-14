import os
import cv2

import torch

import pandas as pd
import numpy as np
from torchvision import transforms

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_meta_test():
    
    meta = pd.read_csv('data/annotations/metadata.csv')
    meta = meta[meta['Split'] == 'TEST']

    meta = meta[meta.apply(
        lambda x: os.path.exists(os.path.join('data/echonet', x['FileName'] + '.avi')),
        axis=1
    )]

    y_trues = meta['EF'].values

    return meta, y_trues


def estimate_L(mask: np.ndarray) -> float:
    
    if np.sum(mask) == 0:
        return 0.0
    
    rows = np.where(np.any(mask == 1, axis=1))[0]
    row_base = rows[0]
    row_apex = rows[-1]
    
    return (row_apex - row_base)


def fbf_prediction(model, device, video_path, pixel_spacing=0.1):

    cap = cv2.VideoCapture(video_path)

    areas = []
    lengths = []

    with torch.no_grad():

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
              break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tensor = image_transforms(gray).unsqueeze(0).float().to(device)
            pred = model(tensor)
            curr_mask = (pred > 0.5).float().cpu().numpy().squeeze().astype(np.uint8)

            areas.append(np.sum(curr_mask))
            lengths.append(estimate_L(curr_mask))

        cap.release()

    return areas, lengths
    

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