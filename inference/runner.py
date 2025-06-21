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

        y_preds = []
        mean_divergence = []
        mean_dice = []

        with tqdm(total=len(meta)) as pbar:
            
            for _, row in meta.iterrows():
                
                file = row['FileName']
                split = row['Split']
                true_ef = row['EF']

                pbar.set_description(f"{file}.avi | True EF: {true_ef:.1f}")

                video_path = os.path.join('data/echonet', file + '.avi')
                
                areas, lengths, flow_, dice_ = fbf_prediction(self.model, self.device, video_path)

                mean_divergence.append(flow_)
                mean_dice.append(dice_)
                
                predicted_ef = estimate_ef(areas, lengths)
                y_preds.append(predicted_ef)

                send(file, split, true_ef, predicted_ef, dice_, flow_, self.model_name)

                pbar.update(1)


        


    
