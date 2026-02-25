# Performance Metrics for Generative Models

This document explains each performance metric used in our Quantum Latent Conditional Flow Matching (QLCFM) experiments. For each metric we describe what it measures, how it is calculated, the original reference, and what values are considered acceptable in top-tier venues (ICML, NeurIPS, CVPR, ICLR).

---

## Table of Contents

1. [MSE (Mean Squared Error)](#1-mse-mean-squared-error)
2. [Reconstruction Loss (Recon)](#2-reconstruction-loss-recon)
3. [KL Divergence (KL)](#3-kl-divergence-kl)
4. [FID (Fréchet Inception Distance)](#4-fid-fréchet-inception-distance)
5. [IS (Inception Score)](#5-is-inception-score)
6. [PSNR (Peak Signal-to-Noise Ratio)](#6-psnr-peak-signal-to-noise-ratio)
7. [SSIM (Structural Similarity Index)](#7-ssim-structural-similarity-index)
8. [LPIPS (Learned Perceptual Image Patch Similarity)](#8-lpips-learned-perceptual-image-patch-similarity)
9. [Summary Table](#9-summary-table)
10. [References](#10-references)

---

## 1. MSE (Mean Squared Error)

### What it measures
MSE quantifies the average squared difference between predicted and target values. In our CFM experiments it is the **velocity-field matching loss**: how well the learned velocity field matches the ground-truth conditional vector field that transports noise to data.

### Formula

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \| \hat{v}(z_t, t) - v_{\text{target}}(z_t, t) \|^2$$

where $\hat{v}$ is the predicted velocity, $v_{\text{target}}$ is the ground-truth conditional velocity field, and the average is over batch samples and time steps.

### Direction
**Lower is better.** MSE = 0 means perfect velocity prediction.

### Reference
- Lipman et al., "Flow Matching for Generative Modeling," ICLR 2023.

### Acceptable thresholds
MSE for velocity-field matching is model- and dataset-specific. There is no universal threshold since it depends on the latent-space dimensionality and normalization. In our experiments:
- Classical baselines (MLP velocity fields): MSE ~ 0.04–0.05
- Quantum velocity fields (v2 best): MSE ~ 0.53
- The key comparison is **relative** MSE between quantum and classical under controlled conditions.

---

## 2. Reconstruction Loss (Recon)

### What it measures
Reconstruction loss quantifies how faithfully the VAE decoder reconstructs the input from its latent representation. It measures pixel-level fidelity of the autoencoder stage.

### Formula

$$\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \| x_i - \hat{x}_i \|^2$$

where $x_i$ is the original image and $\hat{x}_i = \text{Dec}(\text{Enc}(x_i))$ is the reconstruction. Some implementations use binary cross-entropy instead of MSE for normalized pixel values.

### Direction
**Lower is better.** Recon = 0 means perfect reconstruction.

### Reference
- Kingma & Welling, "Auto-Encoding Variational Bayes," ICLR 2014.

### Acceptable thresholds
Reconstruction loss is scale-dependent (depends on image resolution, normalization, pixel range). For CIFAR-10 (32×32, normalized to [0,1]):
- Good VAE reconstruction: Recon < 0.01–0.02 (MSE per pixel)
- Our ResConv VAE: Recon ~ 0.003–0.005 (well-trained)

---

## 3. KL Divergence (KL)

### What it measures
KL divergence measures how much the learned posterior distribution $q(z|x)$ deviates from the prior $p(z) = \mathcal{N}(0, I)$. It serves as a regularizer in VAEs, encouraging the latent space to be smooth and continuous.

### Formula

$$D_{\text{KL}}(q(z|x) \| p(z)) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)$$

where $\mu_j$ and $\sigma_j^2$ are the mean and variance of the $j$-th latent dimension output by the encoder.

### Direction
**Lower is better.** KL = 0 means the posterior exactly matches the prior, but this usually implies posterior collapse (the model ignores the input). A moderate KL indicates a well-structured latent space.

### Reference
- Kingma & Welling, "Auto-Encoding Variational Bayes," ICLR 2014.
- Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework," ICLR 2017.

### Acceptable thresholds
- Typical range for well-trained VAEs: KL ~ 5–50 nats (depends on latent dimension)
- Too low (< 1): Posterior collapse risk — latent codes are uninformative
- Too high (> 100): Under-regularized — latent space may not be smooth
- Our experiments (latent_dim=32): KL ~ 10–30 nats is typical

---

## 4. FID (Fréchet Inception Distance)

### What it measures
FID measures the distance between the distribution of generated images and real images in the feature space of a pre-trained Inception-v3 network. It captures both **fidelity** (quality of individual samples) and **diversity** (coverage of the real data distribution).

### Formula

$$\text{FID} = \| \mu_r - \mu_g \|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

where $(\mu_r, \Sigma_r)$ and $(\mu_g, \Sigma_g)$ are the mean and covariance of the Inception-v3 pool3 features (2048-dimensional) computed from real and generated images, respectively.

### Direction
**Lower is better.** FID = 0 means the generated distribution perfectly matches the real distribution in Inception feature space.

### Standard evaluation protocol
- **50,000 generated samples** compared against **50,000 real samples** (or the full training set)
- Pre-trained Inception-v3 network (ImageNet weights)
- Using fewer samples inflates FID and increases variance — results with < 10,000 samples are unreliable

### Reference
- Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium," NeurIPS 2017.

### Acceptable thresholds (CIFAR-10, unconditional, 50k samples)

| Model class | FID range | Reference |
|---|---|---|
| **Diffusion SOTA** (EDM2, 2024) | ~1.5–2.0 | Karras et al., NeurIPS 2024 |
| **Score-SDE** | ~2.2–2.9 | Song et al., ICLR 2021 |
| **DDPM** | ~3.2 | Ho et al., NeurIPS 2020 |
| **StyleGAN2-ADA** | ~2.9 | Karras et al., NeurIPS 2020 |
| **GANs (typical)** | ~9–30 | Various |
| **VAE (vanilla)** | ~100–220 | Kingma & Welling, 2014 |
| **Flow matching** | ~2–6 | Lipman et al., ICLR 2023 |

**General guidelines for CIFAR-10:**
- FID < 5: State-of-the-art (publishable at top venues as a generative model paper)
- FID < 20: Competitive (acceptable in papers where generation is not the main contribution)
- FID < 50: Reasonable for constrained models (e.g., quantum, low-param)
- FID > 100: Poor — likely mode collapse or severe quality issues

---

## 5. IS (Inception Score)

### What it measures
IS evaluates two properties of generated images:
1. **Quality**: Each generated image should be confidently classified by Inception-v3 (low-entropy class distribution)
2. **Diversity**: The aggregate set of generated images should cover many classes (high-entropy marginal distribution)

### Formula

$$\text{IS} = \exp\!\left(\mathbb{E}_{x \sim p_g} \left[ D_{\text{KL}}(p(y|x) \| p(y)) \right]\right)$$

where $p(y|x)$ is the Inception-v3 softmax output for a single generated image $x$, and $p(y) = \mathbb{E}_{x}[p(y|x)]$ is the marginal class distribution over all generated samples.

### Direction
**Higher is better.** Maximum IS for CIFAR-10 (10 classes) is theoretically 10.0 (every class equally represented, each image perfectly classified).

### Standard evaluation protocol
- **50,000 generated samples**, split into 10 groups, report mean ± std
- Same pre-trained Inception-v3 as FID

### Reference
- Salimans et al., "Improved Techniques for Training GANs," NeurIPS 2016.

### Known limitations
- Does not compare against real data (only evaluates generated samples)
- Biased toward ImageNet-like features
- Can be fooled by mode dropping (generating few but high-quality modes)
- FID is generally preferred as the primary metric; IS is supplementary

### Acceptable thresholds (CIFAR-10, unconditional, 50k samples)

| Model class | IS range | Reference |
|---|---|---|
| **Real CIFAR-10 data** | ~11.24 ± 0.12 | Salimans et al., 2016 |
| **Diffusion SOTA** | ~9.5–10.0 | Various, 2022–2024 |
| **Score-SDE** | ~9.83 | Song et al., ICLR 2021 |
| **DDPM** | ~9.46 | Ho et al., NeurIPS 2020 |
| **GANs (good)** | ~8.5–9.0 | Various |
| **VAE (vanilla)** | ~1.5–2.0 | Kingma & Welling, 2014 |

**General guidelines for CIFAR-10:**
- IS > 9.0: State-of-the-art
- IS > 7.0: Competitive
- IS > 5.0: Reasonable for constrained models
- IS < 3.0: Poor — low quality or diversity

---

## 6. PSNR (Peak Signal-to-Noise Ratio)

### What it measures
PSNR quantifies the ratio between the maximum possible signal power and the noise (reconstruction error) power. It is a standard metric for image reconstruction quality, measuring pixel-level accuracy.

### Formula

$$\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{L^2}{\text{MSE}}\right) \quad \text{dB}$$

where $L$ is the maximum pixel value (255 for 8-bit images, or 1.0 for normalized images) and:

$$\text{MSE} = \frac{1}{H \times W \times C} \sum_{i,j,c} (x_{i,j,c} - \hat{x}_{i,j,c})^2$$

### Direction
**Higher is better.** PSNR → ∞ means perfect reconstruction (MSE → 0).

### Reference
- A classical signal-processing metric, widely used since the 1970s. No single originating paper.
- Commonly cited benchmark: Hore & Ziou, "Image Quality Metrics: PSNR vs. SSIM," ICPR 2010.

### Acceptable thresholds

| Quality level | PSNR (dB) |
|---|---|
| Excellent reconstruction | > 35 dB |
| Good reconstruction | 30–35 dB |
| Acceptable | 25–30 dB |
| Poor | < 25 dB |

**In the context of generative models:**
- PSNR is most meaningful for **reconstruction** tasks (VAE decoder, image super-resolution, inpainting), not for **generation** (where there is no ground-truth target image)
- For VAE reconstruction on CIFAR-10 (32×32): PSNR > 25 dB is acceptable, > 30 dB is good
- Top super-resolution papers (CVPR/ICCV): PSNR improvements of 0.1–0.5 dB are considered significant

### Limitations
- Pixel-level metric: does not account for perceptual quality
- Two images can have similar PSNR but very different perceptual quality
- Should be used alongside perceptual metrics (SSIM, LPIPS)

---

## 7. SSIM (Structural Similarity Index)

### What it measures
SSIM evaluates image quality based on three components: **luminance**, **contrast**, and **structural** similarity. Unlike PSNR, it captures perceptual similarity by comparing local patterns rather than individual pixels.

### Formula

$$\text{SSIM}(x, y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

where:
- $\mu_x, \mu_y$: local means (computed over sliding window)
- $\sigma_x, \sigma_y$: local standard deviations
- $\sigma_{xy}$: local cross-correlation
- $C_1 = (K_1 L)^2$, $C_2 = (K_2 L)^2$: stabilization constants ($K_1 = 0.01$, $K_2 = 0.03$, $L$ = dynamic range)

The final SSIM is averaged over all windows (Mean SSIM, or MSSIM).

### Direction
**Higher is better.** SSIM = 1.0 means perfect structural similarity. Range: [−1, 1], but typically [0, 1] for non-adversarial comparisons.

### Reference
- Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity," IEEE TIP, 2004.

### Acceptable thresholds

| Quality level | SSIM |
|---|---|
| Excellent reconstruction | > 0.95 |
| Good reconstruction | 0.90–0.95 |
| Acceptable | 0.80–0.90 |
| Poor | < 0.80 |

**In the context of generative models:**
- For VAE reconstruction on CIFAR-10: SSIM > 0.85 is acceptable, > 0.90 is good
- Super-resolution papers: SSIM differences of 0.01–0.02 are considered significant
- For low-resolution images (32×32), SSIM tends to be higher than for high-resolution images

---

## 8. LPIPS (Learned Perceptual Image Patch Similarity)

### What it measures
LPIPS measures perceptual distance between two images using deep features from a pre-trained neural network (typically AlexNet or VGG). It correlates more strongly with human perception of image similarity than PSNR or SSIM.

### Formula

$$\text{LPIPS}(x, \hat{x}) = \sum_{\ell} \frac{1}{H_\ell W_\ell} \sum_{h,w} \| w_\ell \odot (\phi_\ell(x)_{h,w} - \phi_\ell(\hat{x})_{h,w}) \|_2^2$$

where:
- $\phi_\ell(x)$: feature activations at layer $\ell$ of a pre-trained network (AlexNet/VGG)
- $w_\ell$: learned per-channel weights calibrated on human perceptual judgments
- The sum is over selected layers $\ell$ and spatial locations $(h, w)$

### Direction
**Lower is better.** LPIPS = 0 means perceptually identical images.

### Reference
- Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric," CVPR 2018.

### Acceptable thresholds

| Quality level | LPIPS |
|---|---|
| Perceptually identical | < 0.05 |
| Good perceptual quality | 0.05–0.10 |
| Acceptable | 0.10–0.20 |
| Noticeable degradation | 0.20–0.40 |
| Poor | > 0.40 |

**In the context of generative models:**
- For VAE reconstruction on CIFAR-10: LPIPS < 0.15 is acceptable, < 0.10 is good
- Image-to-image translation (CVPR/ICCV papers): LPIPS < 0.10–0.15 is typical for good models
- Unlike PSNR/SSIM, LPIPS penalizes blurriness (common in VAEs) more appropriately

---

## 9. Summary Table

| Metric | Direction | Domain | Primary use | Standard protocol |
|---|---|---|---|---|
| **MSE** | Lower ↓ | [0, ∞) | Velocity field matching (CFM training loss) | Per-batch, dataset-specific |
| **Recon** | Lower ↓ | [0, ∞) | VAE reconstruction quality | Per-pixel MSE or BCE |
| **KL** | Lower ↓ (moderate) | [0, ∞) | VAE latent regularization | Per-sample, sum over latent dims |
| **FID** | Lower ↓ | [0, ∞) | Generation quality + diversity | **50k generated vs 50k real** |
| **IS** | Higher ↑ | [1, C] | Generation quality + diversity | **50k generated, 10 splits** |
| **PSNR** | Higher ↑ | [0, ∞) dB | Pixel-level reconstruction | Per-image, average over test set |
| **SSIM** | Higher ↑ | [−1, 1] | Structural reconstruction | Per-image, average over test set |
| **LPIPS** | Lower ↓ | [0, ∞) | Perceptual reconstruction | Per-image, average over test set |

### Metric categories

**Training losses (optimized directly):**
- MSE (CFM velocity matching), Recon (VAE pixel loss), KL (VAE regularization)

**Generation quality (evaluated post-training):**
- FID (primary), IS (supplementary)

**Reconstruction quality (evaluated on VAE):**
- PSNR (pixel-level), SSIM (structural), LPIPS (perceptual)

---

## 10. References

1. **MSE / Flow Matching**: Lipman, Y., Chen, R. T., Ben-Hamu, H., & Nickel, M. (2023). Flow Matching for Generative Modeling. *ICLR 2023*.

2. **Recon / KL / VAE**: Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR 2014*.

3. **FID**: Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. *NeurIPS 2017*.

4. **IS**: Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved Techniques for Training GANs. *NeurIPS 2016*.

5. **PSNR**: Hore, A., & Ziou, D. (2010). Image Quality Metrics: PSNR vs. SSIM. *ICPR 2010*.

6. **SSIM**: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. *IEEE Transactions on Image Processing, 13*(4), 600–612.

7. **LPIPS**: Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR 2018*.

8. **DDPM**: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020*.

9. **Score-SDE**: Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR 2021*.

10. **StyleGAN2-ADA**: Karras, T., Aittala, M., Hellsten, J., Laine, S., Lehtinen, J., & Aila, T. (2020). Training Generative Adversarial Networks with Limited Data. *NeurIPS 2020*.

11. **EDM2**: Karras, T., Aittala, M., Lehtinen, J., Hellsten, J., Laine, S., & Aila, T. (2024). Analyzing and Improving the Training Dynamics of Diffusion Models. *NeurIPS 2024*.

12. **beta-VAE**: Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., & Lerchner, A. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. *ICLR 2017*.
