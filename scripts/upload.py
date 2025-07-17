import wandb

run = wandb.init(project='echonet-dl')

artifact = wandb.Artifact('inference', type='dataset')
artifact.add_file('inference.csv')

run.log_artifact(artifact)

wandb.finish()