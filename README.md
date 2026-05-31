<div align="center">

<h2>DSFM: Functional MRI Time Series Generation via Wavelet-Based Image Transform and Spectral Flow Matching for Brain Disorder Identification</h2>

**_The first framework to introduce dual spectral image transform and spectral flow matching for fMRI BOLD signal generation and brain disorder classification!_**

[Hwa Hui Tew](https://htew0001.github.io/)¹*, [Junn Yong Loo](https://scholar.google.com/citations?user=PsL3CMYAAAAJ&hl=en)¹*, [Fang Yu Leong](https://scholar.google.com/citations?user=RkNP3BAAAAAJ&hl=en)¹, [Hernando Ombao](https://scholar.google.com.my/citations?user=z_BmSq4AAAAJ&hl=en&authuser=1&oi=ao)², [Chee-Ming Ting](https://scholar.google.com.my/citations?user=z_BmSq4AAAAJ&hl=en&authuser=1&oi=ao)¹†

¹ ​School of Information Technology, Monash University Malaysia \
² Statistics Program, King Abdullah University of Science and Technology

[![paper](https://img.shields.io/badge/MICCAI'25-T2IDiff-b31b1b)](https://papers.miccai.org/miccai-2025/paper/3042_paper.pdf)
[![arXiv](https://img.shields.io/badge/ICLR'26-DSFM-b31b1b)](https://openreview.net/pdf?id=Dgphd9qizu)

</div>
<div align="center">
<img src="Images/viz1.png" alt="Visualization">
    
**Figure 1:** Overview of our proposed DSFM framework.
</div>

## 📰 News

- **[2026.05.20]** **Please look forward to our upcoming work!**

- **[2025.05.18]** We have released the code and paper for DSFM !

## 📄 Abstract

Functional Magnetic Resonance Imaging (fMRI) provides non-invasive access to dynamic brain activity by measuring blood oxygen level-dependent (BOLD) signals over time. However, the resource-intensive nature of fMRI acquisition limits the availability of high-fidelity samples required for data-driven brain analysis models. While modern generative models can synthesize fMRI data, they often remain challenging in replicating their inherent non-stationarity, intricate spatiotemporal dynamics, and physiological variations of raw BOLD signals. 

To address these challenges, we propose Dual-Spectral Flow Matching (DSFM), a novel fMRI generative framework that cascades dual frequency representation of BOLD signals with spectral flow matching. Specifically, our framework first converts BOLD signals into a wavelet decomposition map via a discrete wavelet transform (DWT) to capture globalized transient and multi-scale variations, and projects into the discrete cosine transform (DCT) space across brain regions and time to exploit localized energy compaction of low-frequency dominant BOLD coefficients. Subsequently, a spectral flow matching model is trained to generate class-conditioned cosine-frequency representation. The generated samples are reconstructed through inverse DCT and inverse DWT operations to recover physiologically plausible time-domain BOLD signals. This dual-transform approach imposes structured frequency priors and preserves key physiological brain dynamics. Ultimately, we demonstrate the efficacy of our approach through improved downstream fMRI-based brain network classification.

## 🎯 How to Use

### Datasets
#### 1. NetSim 
```
https://www.fmrib.ox.ac.uk/datasets/netsim/index.html
```
#### 2. MDD 
```
https://rfmri.org/REST-meta-MDD
```
After preprocessing, place the processed data in the project’s empty `/data/<desired_dataset>` folder.

### Installation
Download and set up the repository:
```
https://github.com/htew0001/DSFM
cd DSFM
```

We provide a [`requirements.yaml`](requirements.yaml) file to easily create a Conda environment configured to run the model:
```
conda env create -f requirements.yaml
conda activate DSFM
```

### Usage
We include three main scripts to perform different tasks:
- **Conditional Generation**: [`run_training.py`](run_training.py) - Executes the training of conditional generative task for disease and healthy-control groups.
- **Conditional Sampling**: [`run_inference.py`](run_inference.py) - Executes the sampling of conditional generative task for disease and healthy-control groups.
- **Evaluation Metrics**: [`run_evaluation.py`](run_evaluation.py) - Executes the evaluation of various time-series metrics.

**For Training of Conditional Generative Models:**
```
python run_training.py --config ./configs/conditional/<desired_dataset>.yaml
```

**For Sampling of Conditional Generative Models:**
```
python run_inference.py --config ./configs/conditional/<desired_dataset>.yaml
```

**For Evaluation of Conditional Generative Models:**
```
python run_evaluation.py 
```

## ❤️ Acknowledgements

This repo is mainly built on [T2I-Diff](https://github.com/htew0001/T2I-Diff). Thanks for the great work.

## 📝 Citation

If you find our work useful, please cite our related paper:

```
# ICLR 2026
@inproceedings{tewfunctional,
  title={Functional MRI Time Series Generation via Wavelet-Based Image Transform and Spectral Flow Matching for Brain Disorder Identification},
  author={Tew, Hwa Hui and Loo, Junn Yong and Yu, Leong Fang and Lau, Julia K and Fan, Ding and Ombao, Hernando and Phan, Raphael CW and Tan, Chee Pin and Ting, Chee-Ming},
  booktitle={The Fourteenth International Conference on Learning Representations}
}

# MICCAI 2025
@inproceedings{tew2025t2i,
  title={T2I-Diff: fMRI Signal Generation via Time-Frequency Image Transform and Classifier-Free Denoising Diffusion Models},
  author={Tew, Hwa Hui and Loo, Junn Yong and Tan, Yee-Fan and Tang, Xinyu and Ombao, Hernando and Noman, Fuad and Phan, Rapha{\"e}l C-W and Ting, Chee-Ming},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={640--650},
  year={2025},
  organization={Springer}
}
```

## 📧 Contact
For any inquiries, please contact at hwa.tew@monash.edu.
