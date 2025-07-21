import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import joblib
import segmentation_models_pytorch as smp

from utils.inference import get_meta
from runner.runner import InferenceRunner

MODEL_NAME='ResNet50-UNet'
PATH='data/dynamic'

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    )
    model.load_state_dict(torch.load('artifacts/resnet50-unet:v1/resnet50.pth', map_location=torch.device('cpu')))
    bias_predictor = joblib.load('artifacts/e-predictor:v0/gbr.joblib')
    data = get_meta(split='TEST')
    runner = InferenceRunner(model, device, data, PATH, MODEL_NAME, bias_predictor)
    runner.run()