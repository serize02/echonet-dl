import os

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torchvision import transforms
from tqdm import tqdm

from utils.server import send
from utils.inference import fbf_prediction, estimate_ef


class InferenceRunner:

    def __init__(self, model, device, data, path, model_name):
        self.model = model
        self.device = device
        self.data = data
        self.model_name = model_name
        self.path = path

        self.model.to(self.device)
        self.model.eval()

    def run(self):
        meta = self.data
        y_trues = self.data['EF'].values

        with tqdm(total=len(meta)) as pbar:
            for _, row in meta.iterrows():
                file = row['FileName']
                video_path = os.path.join(self.path, file + '.avi')

                pbar.set_description(f"{file}.avi")

                try:
                    stats = estimate_ef(self.model, self.device, video_path)

                    results = {
                        'model_name': self.model_name,
                        'filename': file,
                        'split': row['Split'],
                        'true_ef': row['EF'],
                    }
                    results |= stats

                    send(results)


                except Exception as e:
                    print(f"Skipping {file}.avi due to error: {e}")

                pbar.update(1)