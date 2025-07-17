import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch
import joblib

import segmentation_models_pytorch as smp

from utils.inference import get_meta

from inference.runner import InferenceRunner
from inference import logger

MODEL_NAME='ResNet50-UNet'
INPUT_CHANNELS=1
PATH='data/echonet-dynamic/videos'

if __name__ == '__main__':

    logger.info(f'Running inference for {MODEL_NAME}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')
    
    logger.info(f'loading model ...')
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=INPUT_CHANNELS,
        classes=1
    )
    model.load_state_dict(torch.load('artifacts/resnet50-unet:v0/resnet50-unet.pth', map_location=torch.device('cpu'))) # cpu
    logger.info('model successfully loaded.')
    bias_predictor = joblib.load('artifacts/error_gbr.joblib')
    logger.info('loading bias-predictor ...')
    
    logger.info('bias-predictor successfully loaded.')

    logger.info('loading data ...')
    data = get_meta(split='TEST')
    logger.info('data successfully loaded.')

    logger.info('running ...')

    runner = InferenceRunner(model, device, bias_predictor, data, PATH, MODEL_NAME)
    runner.run()

    logger.info('run finished successfully')