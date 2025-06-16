import requests

def send(filename, true_ef, predicted_ef, dice_stability, flow_divergence, model_name):
    url = "http://localhost:8080/predict"
    
    data = {
        "filename": filename,
        "true_ef": float(true_ef),
        "predicted_ef": float(predicted_ef),
        "dice_stability": float(dice_stability),
        "flow_divergence": float(flow_divergence),
        "model_name": model_name
    }

    try:
        response = requests.post(url, json=data)
        if response.status_code != 200:
            print(f"[ERROR] Failed to send {filename}: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"[EXCEPTION] Could not connect to server: {e}")