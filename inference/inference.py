import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torchvision import transforms
from sklearn.metrics import mean_absolute_error as mae
from PIL import Image
from tqdm import tqdm

from models.unet.unet import UNet

import warnings
warnings.filterwarnings('ignore')

# download W&B artifact

# import wandb
# wandb.login()

# run = wandb.init(project='echonet-dl')
# artifact = run.use_artifact('ernestoserize-constructor-university/echonet-dl/unet:v0', type='model')
# artifact_dir = artifact.download()
# wandb.finish()

INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
# model.load_state_dict(torch.load('artifacts/unet:v0/unet.pth')) # cuda available
model.load_state_dict(torch.load('artifacts/unet:v0/unet.pth', map_location=torch.device('cpu'))) # cpu
model.to(device)
model.eval()

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def estimate_L(mask: np.ndarray) -> float:
        if np.sum(mask) == 0:
            return 0.0
        rows = np.where(np.any(mask == 1, axis=1))[0]
        row_base = rows[0]
        row_apex = rows[-1]
        return (row_apex - row_base)

def fbf_prediction(video_path, pixel_spacing=0.1):

    cap = cv2.VideoCapture(video_path)

    areas = []
    lengths = []

    with torch.no_grad():

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
              break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_tensor = image_transforms(frame_gray).unsqueeze(0).float().to(device)
            pred = model(frame_tensor)
            curr_mask = (pred > 0.5).float().cpu().numpy().squeeze().astype(np.uint8)

            areas.append(np.sum(curr_mask))
            lengths.append(estimate_L(curr_mask))

        cap.release()

    return areas, lengths

def estimate_ef(video_path, pixel_spacing=0.1, denoise=False, threshold=0.005):

    areas, lengths = fbf_prediction(video_path, pixel_spacing)

    volumes = []
    for area_px, length in zip(areas, lengths):
        length_cm = length * pixel_spacing
        area_cm2 = area_px * (pixel_spacing ** 2)
        volume = area_cm2 * length_cm
        volumes.append(volume)

    if denoise:
        frames = len(volumes)
        fft_vals = fft(volumes)
        freqs = fftfreq(frames, 0.01)
        cutoff = threshold * np.max(np.abs(fft_vals))
        fft_filtered = np.where(np.abs(fft_vals) > cutoff, fft_vals, 0)
        volumes = np.real(ifft(fft_filtered))

    edv = max(volumes)
    esv = min(volumes)
    ef = ((edv - esv) / edv) * 100

    return ef, volumes


if __name__ == '__main__':

    meta = pd.read_csv('data/annotations/metadata.csv')
    meta = meta[meta['Split'] == 'TEST']

    meta = meta[meta.apply(
        lambda x: os.path.exists(os.path.join('data/echonet', x['FileName'] + '.avi')),
        axis=1
    )]

    y_trues = meta['EF'].values
    y_preds = []

    for file in tqdm(meta['FileName']):
        ef, _ = estimate_ef(os.path.join('data/echonet', file + '.avi'))
        y_preds.append(ef)

    print(mae(y_trues, y_preds))

