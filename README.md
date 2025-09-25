# DSFM

### Main Contribution
- **Wavelet-Based Image Transform**:  
Generate multivariate BOLD signals to wavelet-based image transform as opposed to univariate time-frequency images.
  
- **Spectral Flow Matching**:  
Introduce spectral flow-matching loss that learns the velocity field of a probability flow directly in the DCT domain, enabling
coarse-to-fine generation aligned with the spectral structure of natural images, while achieving high-fidelity synthesis with significantly fewer sampling steps.

### Settings

```bash
conda create -n dsfm python=3.13.3
conda activate dsfm
```
