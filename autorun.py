import os
import yaml
import csv
import glob
import time
import shutil
import numpy as np
from datetime import datetime
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import cv2
from PIL import Image
import subprocess
import h5py
from scipy.io import savemat

# --- CONFIGURATION ---
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "../input")
output_dir = os.path.join(script_dir, "../output")
log_path = os.path.join(script_dir, "../log.csv")
info_path = os.path.join(script_dir, "info.yaml")

sinsr_input = os.path.join(script_dir, "testdata", "RealSet65")
sinsr_output = os.path.join(script_dir, "results", "SinSR", "RealSet65")

# Create clean I/O folders for SinSR
shutil.rmtree(sinsr_input, ignore_errors=True)
shutil.rmtree(sinsr_output, ignore_errors=True)
os.makedirs(sinsr_input, exist_ok=True)
os.makedirs(sinsr_output, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# --- STEP 1: Read info.yaml ---
with open(info_path, "r") as f:
    info = yaml.safe_load(f)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
scale_factor = int(info.get("scale_factor", 4))
model_info = [timestamp] + [str(v) for v in info.values()]

# --- Helpers ---
def spectral_angle_mapper(ref, est):
    eps = 1e-12
    R = ref.reshape(-1, ref.shape[2]).astype(np.float64)
    E = est.reshape(-1, est.shape[2]).astype(np.float64)
    num = np.sum(R * E, axis=1)
    den = (np.linalg.norm(R, axis=1) * np.linalg.norm(E, axis=1)) + eps
    cosang = np.clip(num / den, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    return float(np.mean(ang))

# --- STEP 2: Collect all .mat files ---
all_results = []
mat_files = glob.glob(os.path.join(input_dir, "*.mat"))

for mat_file in mat_files:
    filename = os.path.splitext(os.path.basename(mat_file))[0]
    print(f"[✓] Processing {filename}.mat")

    try:
        data = {}
        with h5py.File(mat_file, 'r') as f:
            for key in f:
                dset = f[key]
                if isinstance(dset, h5py.Dataset) and dset.ndim == 3:
                    raw = np.array(dset)
                    if raw.shape[0] == 30:
                        cube = np.transpose(raw, (2, 1, 0))  # -> H, W, bands
                    else:
                        cube = raw
                    data[key] = cube
    except Exception as e:
        print(f"[!] Failed to read {mat_file}: {e}")
        continue

    for key in data:
        cube = data[key]
        original_cube = cube.copy()
        H, W, C = cube.shape
        ds_H, ds_W = H // scale_factor, W // scale_factor
        cube_lr = np.zeros((ds_H, ds_W, C), dtype=np.float32)

        for i in range(C):
            band = cv2.resize(cube[:, :, i], (ds_W, ds_H), interpolation=cv2.INTER_CUBIC)
            cube_lr[:, :, i] = band

        band_groups = [[i, i+10, i+20] for i in range(10)]
        for idx, group in enumerate(band_groups):
            try:
                stacked = np.stack([cube_lr[:, :, b] for b in group], axis=-1)
                save_path = os.path.join(sinsr_input, f"{filename}_grp{idx}.png")
                Image.fromarray(np.ascontiguousarray((stacked * 255).astype(np.uint8))).save(save_path)
                print(f"[✓] Saved PNG: {save_path}, shape: {stacked.shape}")
            except Exception as e:
                print(f"[!] Error saving group {idx} for {filename}: {e}")

# --- STEP 3: Run SinSR model on all PNGs ---
print(f"[✓] Running SinSR model on {sinsr_input}")
subprocess.run([
    "python", "inference.py",
    "-i", sinsr_input,
    "-o", sinsr_output,
    "--scale", str(scale_factor),
    "--ckpt", "weights/SinSR_v1.pth",
    "--one_step"
], check=True)

# --- STEP 4: Reconstruct MAT files and compute metrics ---
for mat_file in mat_files:
    filename = os.path.splitext(os.path.basename(mat_file))[0]
    with h5py.File(mat_file, 'r') as f:
        for key in f:
            raw = np.array(f[key])
            if raw.shape[0] == 30:
                original_cube = np.transpose(raw, (2, 1, 0))
            else:
                original_cube = raw

    H, W, C = original_cube.shape
    sr_cube = np.zeros((H, W, C), dtype=np.float32)
    band_groups = [[i, i+10, i+20] for i in range(10)]

    for idx, group in enumerate(band_groups):
        sr_png = os.path.join(sinsr_output, f"{filename}_grp{idx}.png")
        wait_time = 0
        while not (os.path.exists(sr_png) and os.access(sr_png, os.R_OK)):
            time.sleep(0.2)
            wait_time += 0.2
            if wait_time > 10:
                print(f"[!] Timeout waiting for {sr_png}")
                continue
        try:
            img = np.array(Image.open(sr_png)).astype(np.float32) / 255.0
            for j, b in enumerate(group):
                if b < C:
                    sr_cube[:, :, b] = cv2.resize(img[:, :, j], (W, H), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(f"[!] Error loading SR image {sr_png}: {e}")

    # Save reconstructed .mat
    out_path = os.path.join(output_dir, f"{filename}.mat")
    savemat(out_path, {key: sr_cube})

    # Calculate metrics
    psnr_list = [psnr(original_cube[:, :, i], sr_cube[:, :, i], data_range=1.0) for i in range(C)]
    ssim_list = [ssim(original_cube[:, :, i], sr_cube[:, :, i], data_range=1.0) for i in range(C)]
    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))
    avg_sam = spectral_angle_mapper(original_cube, sr_cube)
    print(f"[✓] Metrics for {filename}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, SAM={avg_sam:.4f}")
    all_results.append((avg_psnr, avg_ssim, avg_sam))

# --- STEP 5: Log results ---
try:
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        flat_metrics = []
        for (p, s, a) in all_results:
            flat_metrics.extend([f"{p:.4f}", f"{s:.4f}", f"{a:.4f}"])
        writer.writerow(model_info + flat_metrics)
except PermissionError:
    print(f"[!] Could not write to log file: {log_path}")
