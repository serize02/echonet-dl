import requests
import numpy as np

def send(data):
    url = "http://localhost:8080/predict"

    clean_data = {k: v.item() if isinstance(v, np.generic) else v for k, v in data.items()}

    try:
        response = requests.post(url, json=clean_data)
        if response.status_code != 200:
            print(f"[ERROR] Failed to send {data['filename']}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[EXCEPTION] Could not connect to server: {e}")
