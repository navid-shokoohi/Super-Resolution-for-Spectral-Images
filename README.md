# Hyperspectral Image Super-Resolution – Internship at ImViA Lab

## Overview

This repository was developed as part of my internship project at **ImViA Lab, Université de Bourgogne (Dijon, France)**, under the topic:

> **Image Super-Resolution using Diffusion Models from Synthetic Spectral Data**

The goal of this repository is to provide a reproducible pipeline for benchmarking and automating the evaluation of different **Super-Resolution (SR) models** applied to **Hyperspectral Images (HSIs)**. The code supports:
- Preparing input datasets (MAT files containing hyperspectral cubes)
- Downsampling and grouping spectral bands into RGB-like images
- Running inference with SR models (e.g., SinSR, ESRGAN, SR3, ResShift)
- Reconstructing the output hyperspectral cube
- Calculating metrics (PSNR, SSIM, SAM)
- Logging results into a centralized CSV file for analysis

This pipeline was created for automated experiments required in my internship report, ensuring consistency across models and datasets.

---

## Repository Structure

```
├── autorun.py        # Main automation script (example: SinSR)
├── info.yaml         # Model metadata (name, type, architecture, etc.)
├── input/            # Folder for input .mat files
├── output/           # Folder where reconstructed .mat files are saved
├── log.csv           # Results log (metrics + model info)
├── testdata/         # Temporary folder for PNG inputs for specific models
└── results/          # Folder where model outputs are saved
```

---

## How It Works

### 1. Configuration
- **info.yaml** contains metadata about the model:
  - Name, type, architecture, scale factor, framework, etc.
- `autorun.py` reads this YAML file and includes the information in the log.

### 2. Input Handling
- The script collects all `.mat` files inside the `input/` directory.
- Each file is assumed to contain a hyperspectral cube (`H x W x C`).
- If stored as `(30, H, W)`, it is transposed to `(H, W, 30)`.

### 3. Downsampling & Band Grouping
- Cubes are downsampled according to the `scale_factor` (e.g., ×2 or ×4).
- Spectral bands are grouped into sets of three (e.g., [0,10,20], [1,11,21], …).
- Each group is saved as a PNG image for SR models that operate on 3-channel inputs.

### 4. Running the Model
- The script runs the target SR model (example: **SinSR**) via subprocess.
- Inference is performed on all PNG groups.

### 5. Reconstruction
- Outputs are read back and resized to match the original cube dimensions.
- All groups are merged to reconstruct the hyperspectral cube.
- The result is saved as a `.mat` file in the `output/` folder.

### 6. Metrics
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **SAM** (Spectral Angle Mapper)
- Average values are computed over all bands and logged.

### 7. Logging
- Results are appended to `log.csv` in the following format:

```
timestamp, model_info..., PSNR_1, SSIM_1, SAM_1, PSNR_2, SSIM_2, SAM_2, ...
```

---

## Example Workflow

1. Place `.mat` files into `input/`
2. Adjust `info.yaml` to describe the model
3. Run:
   ```bash
   python autorun.py
   ```
4. Results will be available in:
   - `output/` (reconstructed .mat files)
   - `results/` (model-generated PNGs)
   - `log.csv` (metrics summary)

---

## Internship Context

This repository supports the experimental framework of my internship at **ImViA Lab**, focusing on **Diffusion Models for Hyperspectral Super-Resolution**.  
It enables consistent comparisons across classical (e.g., bicubic), CNN-based, GAN-based, and diffusion-based approaches by unifying the preprocessing, evaluation, and logging steps.

---

## Notes
- Each model may require slight modifications of the script (paths, checkpoints, etc.).
- The provided `autorun.py` example is tailored for **SinSR**.
- Future extensions may include dataset preparation (e.g., MST++), patch generation, and multi-scale evaluation.

---

## License
This repository is for research and academic purposes under the context of my internship report.  
Usage is permitted for reproducibility and further research in hyperspectral image super-resolution.
