import os
import cv2
import torch
import pandas as pd
import numpy as np

from torchvision import transforms
from scipy.spatial.distance import pdist
from scipy.fftpack import dst, idst

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def get_meta(split=None):
    
    meta = pd.read_csv('data/annotations/metadata.csv')
    
    if split is not None:
        meta = meta[meta['Split'] == split]

    meta = meta[meta.apply(
        lambda x: os.path.exists(os.path.join('data/echonet', x['FileName'] + '.avi')),
        axis=1
    )]

    return meta


def estimate_L(mask: np.ndarray) -> float:
    
    if np.sum(mask) == 0:
        return 0.0

    coords = np.column_stack(np.where(mask == 1))
    max_distance = np.max(pdist(coords, metric='euclidean'))

    return float(max_distance)


def stats_helper(arr: list):
    
    if len(arr) > 0:
        mean_arr = float(np.mean(arr))
        var_arr = float(np.var(arr))
        std_arr = float(np.std(arr))
        max_arr = float(np.max(arr))
        min_arr = float(np.min(arr))
    else:
        mean_arr = var_arr = std_arr = max_arr = min_arr = 0.0
    
    return mean_arr, var_arr, std_arr, max_arr, min_arr



def poisson_solver_fft(f):  
    f = f[1:-1, 1:-1]
    n, m = f.shape
    f_sin = dst(dst(f, type=1, axis=0), type=1, axis=1)
    denom = (np.pi * np.arange(1, n+1) / (n+1))[:, None]**2 + (np.pi * np.arange(1, m+1) / (m+1))**2
    u_sin = f_sin / denom
    u = idst(idst(u_sin, type=1, axis=0), type=1, axis=1)
    u /= (4 * (n + 1) * (m + 1))
    return np.pad(u, ((1, 1), (1, 1)), mode='constant')



def flow_stats(masks, pixel_spacing=0.1, alpha=0.8, beta=0.2):

    magnitudes = []
    divergence_scores = []
    vorticity_scores = []
    irrot_energies = []
    soleno_energies = []
    combined_flow_indices = []
    dice_scores = []

    prev = None
    for curr in masks:
        if prev is not None:
            lv_region = curr > 0

            flow = cv2.calcOpticalFlowFarneback(
                prev.astype(np.uint8) * 255,
                curr.astype(np.uint8) * 255,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            vx = flow[..., 0] * pixel_spacing
            vy = flow[..., 1] * pixel_spacing
            
            mag = np.sqrt(vx**2 + vy**2)      
            magnitudes.append(mag[lv_region])
            
            dx = cv2.Sobel(vx, cv2.CV_64F, 1, 0, ksize=5)
            dy = cv2.Sobel(vy, cv2.CV_64F, 0, 1, ksize=5)
            divergence = dx + dy

            avg_divergence = np.mean(divergence[lv_region]) # np.mean(np.abs(divergence[lv_region]))
            divergence_scores.append(avg_divergence)

            dvx_dy = cv2.Sobel(vx, cv2.CV_64F, 0, 1, ksize=5)
            dvy_dx = cv2.Sobel(vy, cv2.CV_64F, 1, 0, ksize=5)
            vorticity = dvy_dx - dvx_dy
            vorticity_scores.append(np.mean(np.abs(vorticity[lv_region])))

            phi = poisson_solver_fft(divergence)
            psi = poisson_solver_fft(vorticity)

            phi_x = cv2.Sobel(phi, cv2.CV_64F, 1, 0, ksize=5)
            phi_y = cv2.Sobel(phi, cv2.CV_64F, 0, 1, ksize=5)
            grad_phi = np.stack([phi_x, phi_y], axis=-1)

            psi_x = cv2.Sobel(psi, cv2.CV_64F, 1, 0, ksize=5)
            psi_y = cv2.Sobel(psi, cv2.CV_64F, 0, 1, ksize=5)
            perp_grad_psi = np.stack([-psi_y, psi_x], axis=-1)

            irrot_energy = 0.5* np.sum((grad_phi[lv_region]**2).sum(axis=-1))
            soleno_energy = 0.5 * np.sum((perp_grad_psi[lv_region]**2).sum(axis=-1))
            irrot_energies.append(irrot_energy)
            soleno_energies.append(soleno_energy)

            e_tot = irrot_energy + soleno_energy
            M = alpha * (irrot_energy / e_tot) + beta * (soleno_energy / e_tot) if e_tot > 0.0 else 0.0
            combined_flow_indices.append(M)

            intersection = np.logical_and(prev, curr).sum()
            dice = (2.0 * intersection) / (prev.sum() + curr.sum() + 1e-5)
            dice_scores.append(dice)

        prev = curr

    mags = np.concatenate(magnitudes) if magnitudes else np.array([])

    mean_mag, var_mag, std_mag, max_mag, min_mag = stats_helper(mags)
    mean_div, var_div, std_div, max_div, min_div = stats_helper(divergence_scores)
    mean_vor, var_vor, std_vor, max_vor, min_vor = stats_helper(vorticity_scores)
    mean_irrot, var_irrot, std_irrot, max_irrot, min_irrot = stats_helper(irrot_energies)
    mean_soleno, var_soleno, std_soleno, max_soleno, min_soleno = stats_helper(soleno_energies)
    mean_combined_flow_index, var_combined_flow_index, std_combined_flow_index, max_combined_flow_index, min_combined_flow_index = stats_helper(combined_flow_indices)
    mean_dice, var_dice, std_dice, max_dice, min_dice = stats_helper(dice_scores)

    stats = {
        'magnitude_mean': mean_mag,
        'magnitude_var': var_mag,
        'magnitude_std': std_mag,
        'magnitude_range': max_mag - min_mag,

        'divergence_mean': mean_div,
        'divergence_var': var_div,
        'divergence_std': std_div,
        'divergence_range': max_div - min_div,

        'vorticity_mean': mean_vor,
        'vorticity_var': var_vor,
        'vorticity_std': std_vor,
        'vorticity_range': max_vor - min_vor,

        'irrot_energy_mean': mean_irrot,
        'irrot_energy_var': var_irrot,
        'irrot_energy_std': std_irrot,
        'irrot_energy_range': max_irrot - min_irrot,

        'soleno_energy_mean': mean_soleno,
        'soleno_energy_var': var_soleno,
        'soleno_energy_std': std_soleno,
        'soleno_energy_range': max_soleno - min_soleno,

        'combined_flow_index_mean': mean_combined_flow_index,
        'combined_flow_index_var': var_combined_flow_index,
        'combined_flow_index_std': std_combined_flow_index,
        'combined_flow_index_range': max_combined_flow_index - min_combined_flow_index,

        'dice_mean': mean_dice,
        'dice_var': var_dice,
        'dice_std': std_dice,
        'dice_range': max_dice - min_dice,
    }

    return stats



def fbf_prediction(model, device, video_path, pixel_spacing=0.1, max_frames=100):

    cap = cv2.VideoCapture(video_path)

    masks = []
    areas = []
    lengths = []

    frames = 0

    with torch.no_grad():

        while cap.isOpened() and frames < max_frames:

            ret, frame = cap.read()
            if not ret:
              break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tensor = image_transforms(gray).unsqueeze(0).float().to(device)
            pred = model(tensor)
            curr_mask = (pred > 0.5).float().cpu().numpy().squeeze().astype(np.uint8)

            masks.append(curr_mask)
            areas.append(np.sum(curr_mask))
            lengths.append(estimate_L(curr_mask))

            frames += 1

        cap.release()

    return areas, lengths, flow_stats(masks)
    

def estimate_ef(model, device, video_path, pixel_spacing=0.1):
    
    areas, lengths, flow_stats = fbf_prediction(model, device, video_path)

    volumes = []

    for area_px, length in zip(areas, lengths):
        length_cm = length * pixel_spacing
        area_cm2 = area_px * (pixel_spacing ** 2)
        volume = (5 / 6) * area_cm2 * length_cm
        volumes.append(volume)

    volumes = np.array(volumes)

    volume_mean, volume_var, volume_std, volume_max, volume_min = stats_helper(volumes)
    length_mean, length_var, length_std, length_max, length_min = stats_helper(lengths)
    area_mean, area_var, area_std, area_max, area_min = stats_helper(areas)

    stats = {
        'predicted_ef': ((volume_max - volume_min) / volume_max) * 100 if volume_max > 0 else 0.0,
        
        'volume_mean': volume_mean,
        'volume_var': volume_var,
        'volume_std': volume_std,
        'volume_range': volume_max - volume_min,
        'volume_ratio': volume_min / volume_max if volume_max > 0 else 0.0,

        'length_mean': length_mean,
        'length_std': length_std,
        'length_range': length_max - length_min,
        'length_ratio': length_min / length_max if length_max > 0 else 0.0,

        'area_mean': area_mean,
        'area_std': area_std,
        'area_range': area_max - area_min,
        'area_ratio': area_min / area_max if area_max > 0 else 0.0
    }

    return stats | flow_stats