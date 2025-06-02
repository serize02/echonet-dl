import wandb

run = wandb.init(
    project='echonet-dl',
    job_type='download model',
    config={'model': 'unet:v1'} 
)

artifact = run.use_artifact('ernestoserize-constructor-university/echonet-dl/unet:v1', type='model')
artifact_dir = artifact.download()

wandb.finish()