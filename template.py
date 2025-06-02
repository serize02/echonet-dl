import os
    
directories = [
    '.github/workflows',
    'config',
    'data/raw',
    'data/processed',
    'data/annotations',
    'notebooks',
    'models',
    'denoise'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

files = {
    "Dockerfile": "# Placeholder for backend Dockerfile\n",
    "docker-compose.local.yml": "# Placeholder for Docker Compose config\n",
    "config/config.yaml": "# Global config settings\n",
}

for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)
    print(f"Created file: {path}")