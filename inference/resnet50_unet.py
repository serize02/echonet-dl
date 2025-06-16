import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np

from sklearn.metrics import mean_absolute_error as mae
from tqdm import tqdm

import segmentation_models_pytorch as smp
from utils.server import send
from utils.inference import (image_transforms,
                            fbf_prediction,
                            get_meta_test,
                            estimate_ef)

MODEL_NAME = 'ResNet50-UNet'
INPUT_CHANNELS = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=INPUT_CHANNELS,
    classes=1
)

# model.load_state_dict(torch.load('artifacts/resnet50-unet:v0/resnet50-unet.pth')) # cuda available
model.load_state_dict(torch.load('artifacts/resnet50-unet:v0/resnet50-unet.pth', map_location=torch.device('cpu'))) # cpu
model.to(device)
model.eval()

if __name__ == '__main__':

    meta, y_trues = get_meta_test()
    
    y_preds = []
    flow_divergence = []
    dices = []

    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        
        file = row['FileName']
        true_ef = row['EF']
        video_path = os.path.join('data/echonet', file + '.avi')
        areas, lengths, flow_, dice_ = fbf_prediction(model, device, video_path)
        flow_divergence.append(flow_)
        dices.append(dice_)
        predicted_ef = estimate_ef(areas, lengths)
        y_preds.append(predicted_ef)

        send(file, true_ef, predicted_ef, dice_, flow_, MODEL_NAME)

    print(f"MAE: {mae(y_trues, y_preds)}")
    print(f"Mean Flow Stability: {np.mean(flow_divergence)}")
    print(f"Mean Dice Stability: {np.mean(dices)}")