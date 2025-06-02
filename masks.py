import os
import cv2
import numpy as np
import pandas as pd

from utils.common import read_yaml
from tqdm import tqdm
from pathlib import Path

config = read_yaml(Path('config/config.yaml'))['data']

tracings = pd.read_csv(config['tracings'])
metadata = pd.read_csv(config['metadata'])
echonet = config['echonet']
raw = config['raw']

grouped = tracings.groupby(['FileName', 'Frame'])

sp = ['train', 'val', 'test']

for s in sp:
    os.makedirs(f'{raw}/{s}/images', exist_ok=True)
    os.makedirs(f'{raw}/{s}/masks', exist_ok=True)

for (filename, frame_num), group in tqdm(grouped, desc='frame-mask-extraction'):

    video_path = f'{echonet}/{filename}'
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)

    ret, frame = cap.read()

    if not ret:
        print(f"Failed to read frame {frame_num} from {filename}")
        continue

    sp = str.lower(metadata[metadata['FileName'] == filename[:-4]]['Split'].values[0])

    image_path = f'{raw}/{sp}/images/{filename}_frame{frame_num}.png'

    cv2.imwrite(image_path, frame)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    points = group[['X', 'Y']].values.astype(int)
    cv2.fillPoly(mask, [points], color=255)

    mask_path = f'{raw}/{sp}/masks/{filename}_frame{frame_num}.png'
    cv2.imwrite(mask_path, mask)

    cap.release()