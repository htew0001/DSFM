# DSFM

## Main Contributions
- **Wavelet-Based Image Transform**:  
Generate multivariate BOLD signals using wavelet-based image transform as opposed to univariate and fixed-resolution time-frequency images.
  
- **Spectral Flow Matching**:  
Introduce spectral flow-matching loss that learns the velocity field of a probability flow directly in the DCT domain, enabling
coarse-to-fine generation aligned with the spectral structure of natural images, while achieving high-fidelity synthesis with significantly fewer sampling steps.

## Datasets Overview
1. **NetSim (Simulated)**
```bash
https://www.fmrib.ox.ac.uk/datasets/netsim/index.html
```
2. **MDD (Real)**
```bash
https://rfmri.org/REST-meta-MDD
```

## Settings
1. **Conda environment Setup**:
```bash
conda create -n dsfm python=3.13.3
conda activate dsfm
```

2. **Install PyTorch + CUDA 12.8**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```


4. **Core libraries**:
```bash
pip install accelerate
pip install -U xformers --no-build-isolation
pip install absl-py ml_collections einops wandb ftfy transformers
pip install opencv-python
pip install scipy
pip install matplotlib
pip install torchdiffeq
```

5. **Running**
```bash
cd dsfm
accelerate launch --mixed_precision fp16 train_ode_dct_uncond.py --config=configs/fmri_diffts24_uvit_mid_2by2_ode_uncond.py --workdir=YOUR_PATH
```  

6. **FID evaluation**
```bash
accelerate launch --mixed_precision fp16 eval.py --config=configs/fmri_diffts24_uvit_mid_2by2_ode_uncond.py --output_path=fmri_ode_block_reweight/FID --nnet_path=fmri_ode_block_reweight/ckpts/140000.ckpt/nnet_ema.pth
```  
