import wandb

run = wandb.init(project='echonet-dl')
artifact = run.use_artifact('ernestoserize-constructor-university/echonet-dl/resnet50-unet:v1', type='model')
artifact_dir = artifact.download()
wandb.finish()