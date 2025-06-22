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
    
    def __init__(self, model, device, data, model_name):
        
        self.model = model
        self.device = device
        self.data = data
        self.model_name = model_name

        self.model.to(self.device)
        self.model.eval()

    def run(self):

        meta = self.data
        y_trues = self.data['EF'].values

        with tqdm(total=len(meta)) as pbar:
            
            for _, row in meta.iterrows():
                
                file = row['FileName']
                
                results = {
                    'model_name': self.model_name,
                    'filename': file,
                    'split': row['Split'],
                    'true_ef': row['EF']
                }

                pbar.set_description(f"{file}.avi")

                video_path = os.path.join('data/echonet', file + '.avi')
                
                stats = estimate_ef(self.model, self.device, video_path)

                send(results|stats)

                pbar.update(1)



        


    
