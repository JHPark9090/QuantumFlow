# VAE v3: SOTA Architecture for Quantum Latent Conditional Flow Matching

## 1. Motivation

The VAE v3 is a major architectural upgrade designed to address the dominant bottleneck in our Quantum Latent Conditional Flow Matching (QLCFM) pipeline: **reconstruction quality**. Prior experiments with VAE v2 showed that the VAE encoder/decoder quality is the single most important factor determining generation FID, far outweighing the choice of velocity field (quantum vs. classical). VAE v2 achieved PSNR 17.59 dB, SSIM 0.40, and recon FID 215 on CIFAR-10 — well below publication standards for top-tier venues (PSNR 25-30+, SSIM 0.80+, FID <50).

VAE v3 incorporates state-of-the-art techniques from the Stable Diffusion VAE (Rombach et al., 2022), VQGAN (Esser et al., 2021), LSGM (Vahdat et al., 2021), and modern perceptual loss literature (Zhang et al., 2018) to maximize reconstruction fidelity while preserving compatibility with the flat-latent CFM pipeline.

## 2. Key Upgrades over VAE v2

| Component | VAE v2 | VAE v3 |
|-----------|--------|--------|
| **Normalization** | BatchNorm | GroupNorm(32) |
| **Activation** | ReLU | SiLU (Swish) |
| **Channel widths** | 32 &rarr; 64 &rarr; 128 &rarr; 256 | 64 &rarr; 128 &rarr; 256 &rarr; 256 |
| **Self-attention** | None | At 8&times;8 and 4&times;4 resolutions |
| **Output activation** | Sigmoid [0, 1] | Tanh [-1, 1] |
| **Reconstruction loss** | MSE | L1 |
| **Perceptual loss** | VGG feature matching (L1) | LPIPS (Zhang et al., 2018) |
| **Adversarial loss** | None | PatchGAN with hinge loss |
| **Weight averaging** | None | EMA (decay = 0.999) |
| **Total parameters** | ~2.1M | ~13.5M |
| **Latent dimension** | 32 | 64 |

## 3. Architecture

### 3.1 ResBlock_v3

The fundamental building block. Each residual block follows the pre-activation design used in the Stable Diffusion VAE (Rombach et al., 2022):

```
Input (C_in)
  |
  +---> GroupNorm(32, C_in) -> SiLU -> Conv2d(C_in, C_out, 3x3) ->
  |     GroupNorm(32, C_out) -> SiLU -> Conv2d(C_out, C_out, 3x3)
  |                                                    |
  +--- [1x1 conv if C_in != C_out, else Identity] ----+
  |                                                    |
  +--------------------( + )---------------------------+
  |
Output (C_out)
```

**Design choices:**
- **GroupNorm** (Wu & He, 2018) with 32 groups instead of BatchNorm. GroupNorm is independent of batch size, which is critical for adversarial training where smaller batches are preferred for stability.
- **SiLU** (Swish, Ramachandran et al., 2017): `x * sigmoid(x)`. Smoother gradient flow than ReLU. Used by Stable Diffusion, DALL-E 2, and most modern generative architectures.
- **Pre-activation order** (He et al., 2016): Norm &rarr; Activation &rarr; Conv, rather than Conv &rarr; Norm &rarr; Activation. Provides better gradient propagation in deep residual networks.
- **Skip connection with optional 1&times;1 conv** for channel dimension changes.

### 3.2 Self-Attention

Single-head self-attention applied at 8&times;8 and 4&times;4 spatial resolutions (following VQGAN and Stable Diffusion):

```
Input x (B, C, H, W)
  |
  +--> GroupNorm(32, C) --> h
  |
  +--> Q = Conv1x1(h)   shape: (B, C, H*W)
  +--> K = Conv1x1(h)   shape: (B, C, H*W)
  +--> V = Conv1x1(h)   shape: (B, C, H*W)
  |
  +--> Attn = softmax(Q^T K / sqrt(C))   shape: (B, H*W, H*W)
  |
  +--> Out = V @ Attn^T   shape: (B, C, H*W) -> reshape (B, C, H, W)
  |
  +--> proj = Conv1x1(Out)
  |
  x + proj   (residual connection)
```

**Why self-attention?** Convolutional layers have limited receptive fields. At 8&times;8 (64 spatial positions) and 4&times;4 (16 positions), self-attention captures long-range spatial dependencies at manageable computational cost. This is the same strategy used by VQGAN (Esser et al., 2021) and the Stable Diffusion autoencoder (Rombach et al., 2022). Attention at higher resolutions (e.g., 32&times;32) would be prohibitively expensive: O(1024^2) attention matrix.

**Single-head** attention is sufficient at these small spatial resolutions (Esser et al., 2021). Multi-head attention adds negligible benefit at 4&times;4.

### 3.3 Encoder

The encoder progressively downsamples spatial resolution from 32&times;32 to 2&times;2 while increasing channel depth, then projects to a flat latent vector:

```
Layer                                           Resolution   Channels
-----                                           ----------   --------
Conv2d(3, 64, 3x3, stride=1, pad=1)             32x32        64
ResBlock_v3(64)                                  32x32        64
ResBlock_v3(64)                                  32x32        64
Conv2d(64, 64, 3x3, stride=2, pad=1)            16x16        64       [downsample]
ResBlock_v3(64 -> 128)                           16x16        128
ResBlock_v3(128)                                 16x16        128
Conv2d(128, 128, 3x3, stride=2, pad=1)          8x8          128      [downsample]
ResBlock_v3(128 -> 256)                          8x8          256
ResBlock_v3(256)                                 8x8          256
SelfAttention(256)                               8x8          256      [global context]
Conv2d(256, 256, 3x3, stride=2, pad=1)          4x4          256      [downsample]
ResBlock_v3(256)                                 4x4          256
ResBlock_v3(256)                                 4x4          256
SelfAttention(256)                               4x4          256      [global context]
GroupNorm(32, 256) + SiLU                        4x4          256
Conv2d(256, 256, 3x3, stride=2, pad=1)          2x2          256      [downsample]
---------------------------------------------------------------------------
Flatten: 256 x 2 x 2 = 1024
fc_mu:     Linear(1024, 64)                                            [mean]
fc_logvar: Linear(1024, 64)                                            [log-variance]
```

**Downsampling strategy:** Strided convolutions (stride=2) instead of max-pooling. Learnable downsampling preserves more information (Radford et al., 2016; Isola et al., 2017).

**Logvar clamping:** `torch.clamp(logvar, -20, 2)` prevents numerical instability. This bounds the standard deviation to [exp(-10), exp(1)] = [4.5e-5, 2.72].

### 3.4 Decoder

The decoder mirrors the encoder, using transposed convolutions for upsampling:

```
Layer                                           Resolution   Channels
-----                                           ----------   --------
Linear(64, 1024) + SiLU                         (flat)       1024
Reshape                                          2x2          256
ConvTranspose2d(256, 256, 4x4, stride=2, pad=1) 4x4          256      [upsample]
ResBlock_v3(256)                                 4x4          256
ResBlock_v3(256)                                 4x4          256
SelfAttention(256)                               4x4          256      [global context]
ConvTranspose2d(256, 256, 4x4, stride=2, pad=1) 8x8          256      [upsample]
ResBlock_v3(256 -> 128)                          8x8          128
ResBlock_v3(128)                                 8x8          128
SelfAttention(128)                               8x8          128      [global context]
ConvTranspose2d(128, 128, 4x4, stride=2, pad=1) 16x16        128      [upsample]
ResBlock_v3(128 -> 64)                           16x16        64
ResBlock_v3(64)                                  16x16        64
ConvTranspose2d(64, 64, 4x4, stride=2, pad=1)   32x32        64       [upsample]
ResBlock_v3(64)                                  32x32        64
ResBlock_v3(64)                                  32x32        64
GroupNorm(32, 64) + SiLU                         32x32        64
Conv2d(64, 3, 3x3, pad=1)                       32x32        3
Tanh                                             32x32        3        [output: [-1, 1]]
```

**Tanh output:** Images are normalized to [-1, 1]. Tanh naturally constrains the output range without hard clipping, unlike Sigmoid which compresses gradients near 0 and 1 (Karras et al., 2020; Rombach et al., 2022). This is the standard in modern generative models (StyleGAN, Stable Diffusion, DALL-E).

**SiLU before fc_dec reshape:** The `fc_dec` Linear layer output passes through SiLU before reshaping to (B, 256, 2, 2), adding nonlinearity at the latent-to-spatial boundary.

### 3.5 Parameter Count

| Component | Parameters |
|-----------|-----------|
| Encoder convolutions + ResBlocks | ~5.8M |
| Encoder self-attention (2 layers) | ~0.5M |
| Encoder FC (mu + logvar) | ~131K |
| Decoder FC | ~66K |
| Decoder convolutions + ResBlocks | ~6.4M |
| Decoder self-attention (2 layers) | ~0.4M |
| Decoder output conv + Tanh | ~1.7K |
| **Total VAE** | **~13.5M** |
| PatchGAN Discriminator | ~663K |

## 4. PatchGAN Discriminator

The discriminator follows the PatchGAN architecture (Isola et al., 2017; Esser et al., 2021) that classifies overlapping patches of the image as real or fake:

```
Conv2d(3, 64, 4x4, stride=2, pad=1)     16x16    LeakyReLU(0.2)
Conv2d(64, 128, 4x4, stride=2, pad=1)   8x8      GroupNorm(32) + LeakyReLU(0.2)
Conv2d(128, 256, 4x4, stride=2, pad=1)  4x4      GroupNorm(32) + LeakyReLU(0.2)
Conv2d(256, 1, 4x4, stride=1, pad=1)    3x3      [patch predictions]
```

Output: (B, 1, 3, 3) — each spatial location classifies a receptive field patch. The PatchGAN encourages high-frequency sharpness that pixel-wise losses (L1/MSE) cannot capture. It penalizes blurriness locally rather than globally (Isola et al., 2017).

**No spectral normalization** is applied; the R1 gradient penalty (Section 5.5) provides sufficient training stability at this scale.

## 5. Loss Functions

### 5.1 Reconstruction Loss: L1

```
L_recon = mean(|x_hat - x|)
```

L1 loss (mean absolute error) produces sharper reconstructions than MSE/L2, which tends to average over modes and produce blurry outputs (Isola et al., 2017; Zhao et al., 2017). For images in [-1, 1], L1 directly penalizes per-pixel deviation.

### 5.2 KL Divergence with Free Bits

```
KL_per_dim = -0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
KL_clamped = max(KL_per_dim_mean, free_bits)     [per dimension, averaged over batch]
KL = sum(KL_clamped)                               [sum over latent dimensions]
```

The **free bits** mechanism (Kingma et al., 2016) sets a minimum KL contribution of 0.25 nats per latent dimension. This prevents **posterior collapse** — a failure mode where the encoder ignores the input and the decoder generates from pure noise. Each latent dimension must carry at least 0.25 nats of information.

**Beta-warmup:** The KL weight `beta` (default 0.001) is linearly ramped from 0 to its target value over 10 epochs. This gives the reconstruction pathway time to converge before the KL regularization activates (Bowman et al., 2016; Sonderby et al., 2016).

### 5.3 LPIPS Perceptual Loss

```
L_lpips = LPIPS(x_hat, x)     [VGG backbone]
```

LPIPS (Learned Perceptual Image Patch Similarity, Zhang et al., 2018) measures perceptual distance using features from a pre-trained VGG network. Unlike raw VGG feature matching (used in VAE v2), LPIPS uses learned per-channel weights calibrated on human perceptual judgments. It is the de facto standard for perceptual quality in generative models.

Weight: `lambda_lpips = 1.0`. The LPIPS loss operates in [-1, 1] input range, matching our data normalization.

### 5.4 Adversarial Hinge Loss

**Discriminator loss:**
```
L_D = mean(max(0, 1 - D(x_real))) + mean(max(0, 1 + D(x_fake)))
```

**Generator loss:**
```
L_G_adv = -mean(D(x_fake))
```

The **hinge loss** (Lim & Ye, 2017; Miyato et al., 2018; Brock et al., 2019) is more stable than the original GAN minimax loss or Wasserstein loss. It saturates once the discriminator is confident, preventing gradient explosion. This is the same formulation used in VQGAN (Esser et al., 2021) and BigGAN (Brock et al., 2019).

Weight: `lambda_adv = 0.1` (base value). When adaptive adversarial weighting is enabled (Section 5.6), this is modulated by the gradient ratio.

### 5.5 R1 Gradient Penalty

```
R1 = E[ || grad_x D(x_real) ||^2 ]
L_D = L_D_hinge + (gamma / 2) * R1
```

The **R1 gradient penalty** (Mescheder et al., 2018) penalizes the discriminator's gradient magnitude on real images. This prevents the discriminator from becoming overly confident, which would cause it to provide uninformative gradients to the generator. Without R1, we observed the discriminator loss approaching ~2.0 (the hinge loss saturation point) within 20 epochs of adversarial training, causing PSNR regression.

**Lazy regularization** (Karras et al., 2020): R1 is computed every 16 batches instead of every batch, with the penalty scaled by the interval (`gamma/2 * R1 * r1_every`). This amortizes the cost of the extra backward pass through the discriminator while preserving the regularization effect. This is the same strategy used in StyleGAN2.

Parameters: `gamma = 10.0`, `r1_every = 16`.

### 5.6 Adaptive Adversarial Weight

Following VQGAN (Esser et al., 2021, Eq. 7):

```
lambda_adaptive = || grad_L_nll w.r.t. psi || / (|| grad_L_adv w.r.t. psi || + 1e-6)
```

where `psi` is the last convolutional layer of the decoder (`Conv2d(64, 3, 3x3)`) and `L_nll` is the combined reconstruction + perceptual loss (following the VQGAN codebase convention). The `1e-6` term prevents division by zero.

This **adaptive weight** automatically balances reconstruction and adversarial gradients at the decoder output. If the adversarial loss produces much larger gradients than the reconstruction loss, the adaptive weight scales it down (and vice versa). This prevents the adversarial signal from overwhelming reconstruction quality — exactly the failure mode we observed in initial training where PSNR dropped from 19.55 to 18.23 dB after enabling the discriminator.

The adaptive weight is clamped to [0, 10000] and detached from the computation graph.

### 5.7 Total Loss

**Stage 1 (Epochs 1-50, warmup):**
```
L_total = L_recon + beta_eff * KL + lambda_lpips * L_lpips
```

**Stage 2 (Epochs 51-300, adversarial):**
```
L_G = L_recon + beta_eff * KL + lambda_lpips * L_lpips + lambda_adaptive * L_G_adv
L_D = L_D_hinge + (gamma / 2) * R1     [R1 every 16 batches]
```

The two-stage approach follows VQGAN (Esser et al., 2021): the VAE first learns stable reconstructions, then the discriminator is introduced to sharpen outputs. Starting adversarial training from epoch 1 often leads to training collapse.

## 6. Training Details

### 6.1 Optimizers

- **Generator (VAE):** Adam, lr=1e-4, betas=(0.5, 0.9)
- **Discriminator:** Adam, lr=4e-4, betas=(0.5, 0.9)

The lower beta_1=0.5 (vs. the default 0.9) is standard for adversarial training — it reduces momentum, preventing the optimizer from overshooting in the adversarial min-max game (Radford et al., 2016; Miyato et al., 2018). The discriminator uses a higher learning rate (4&times;) following the TTUR (Two-Time-scale Update Rule, Heusel et al., 2017).

### 6.2 Learning Rate Schedule

Cosine annealing over 300 epochs with minimum lr=1e-6 for both generator and discriminator:

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T_max))
```

### 6.3 Exponential Moving Average (EMA)

```
theta_EMA <- decay * theta_EMA + (1 - decay) * theta
```

Decay = 0.999. EMA smooths weight updates over ~1000 steps, reducing noise and improving generation quality at inference time. This is standard practice in diffusion models (Ho et al., 2020; Song et al., 2021) and VQGAN. The EMA weights are used for all evaluation and the final saved model.

### 6.4 Data Preprocessing

Images are normalized to [-1, 1]: `x = x * 2 - 1` (from [0, 1] after dividing by 255). This matches the Tanh output range. For FID/IS computation downstream, generated images are rescaled back to [0, 1]: `x_01 = (x + 1) / 2`.

### 6.5 LPIPS Sampling

Computing LPIPS every batch is expensive (VGG forward pass per batch). To reduce epoch time without sacrificing perceptual loss guidance, LPIPS is computed every `lpips_every` batches (default: 4). In batches where LPIPS is skipped, the loss consists of L1 + KL (+ adversarial if active). This reduces epoch time by approximately 3-4&times; with negligible impact on final quality, since LPIPS gradients are relatively smooth across consecutive batches.

### 6.6 Training Schedule Summary

| Phase | Epochs | Loss Components | Description |
|-------|--------|----------------|-------------|
| Beta warmup | 1-10 | L1 + beta(ramp)*KL + LPIPS | KL weight linearly ramped |
| Reconstruction | 11-50 | L1 + beta*KL + LPIPS | Full reconstruction training |
| Adversarial | 51-300 | L1 + beta*KL + LPIPS + adv + R1 | PatchGAN with R1 penalty and adaptive weight |

Total walltime: ~12-18 hours on 1 NVIDIA A100 (80GB HBM).

## 7. Evaluation Metrics

| Metric | Description | Data Range |
|--------|-------------|------------|
| **PSNR** | Peak Signal-to-Noise Ratio | data_range=2.0 (for [-1,1]) |
| **SSIM** | Structural Similarity Index | data_range=2.0 |
| **LPIPS** | Learned Perceptual Image Patch Similarity | [-1, 1] input (SqueezeNet backbone for eval) |
| **Recon FID** | FID between real test images and their reconstructions | [0, 1] after rescaling |

Model selection uses **best validation PSNR** with EMA weights.

## 8. Integration with the CFM Pipeline

The VAE v3 is used as Phase 1 (encoder/decoder) in the two-phase QLCFM pipeline:

```
Phase 1: VAE v3 Training (this file)
    CIFAR-10 images (32x32x3) -> Encoder -> latent z (dim=64) -> Decoder -> reconstruction

Phase 2: CFM Training (QuantumLatentCFM_vaev3.py)
    Frozen VAE encoder maps images to latent space
    Velocity field (quantum or classical) learns flow matching in latent space
    z(t) = (1-t) * z_noise + t * z_data   [OT interpolation]
    v_theta(z(t), t) learns to predict z_data - z_noise

Generation:
    z(0) ~ N(0, I)   [sample noise]
    ODE solve: dz/dt = v_theta(z(t), t), t: 0 -> 1
    z(1) -> Frozen VAE decoder -> generated image
    Rescale: image_01 = (image_tanh + 1) / 2
```

The VAE interface is preserved: `.encode(x) -> (mu, logvar)` and `.decode(z) -> x_hat`. The only adaptation for CFM is the [-1, 1] to [0, 1] rescaling when computing FID/IS on generated images.

## 9. References

1. **Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B.** (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022.* — Stable Diffusion VAE architecture (GroupNorm + SiLU + self-attention, Tanh output).

2. **Esser, P., Rombach, R., & Ommer, B.** (2021). Taming Transformers for High-Resolution Image Synthesis. *CVPR 2021.* — VQGAN: PatchGAN discriminator with hinge loss, two-stage training, self-attention in autoencoder.

3. **Vahdat, A., & Kautz, J.** (2020). NVAE: A Deep Hierarchical Variational Autoencoder. *NeurIPS 2020.* — Deep hierarchical VAE with residual cells and spectral regularization.

4. **Vahdat, A., Kreis, K., & Kautz, J.** (2021). Score-based Generative Modeling in Latent Space. *NeurIPS 2021.* — LSGM: Latent score-based generative model, demonstrating the value of high-quality VAE encoders for latent-space generation.

5. **Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O.** (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR 2018.* — LPIPS perceptual loss.

6. **Wu, Y. & He, K.** (2018). Group Normalization. *ECCV 2018.* — GroupNorm as a batch-size-independent alternative to BatchNorm.

7. **Ramachandran, P., Zoph, B., & Le, Q. V.** (2017). Searching for Activation Functions. *arXiv:1710.05941.* — SiLU/Swish activation function.

8. **Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A.** (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR 2017.* — PatchGAN discriminator, L1 + adversarial loss combination.

9. **Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., & Welling, M.** (2016). Improved Variational Inference with Inverse Autoregressive Flow. *NeurIPS 2016.* — Free bits for preventing posterior collapse.

10. **Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., & Bengio, S.** (2016). Generating Sentences from a Continuous Space. *CoNLL 2016.* — KL annealing / beta warmup for VAE training.

11. **Lim, J. H. & Ye, J. C.** (2017). Geometric GAN. *arXiv:1705.02894.* — Hinge loss for GANs.

12. **Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y.** (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR 2018.* — Spectral normalization and hinge loss.

13. **Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S.** (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. *NeurIPS 2017.* — Two-Time-scale Update Rule (TTUR) for discriminator/generator learning rates.

14. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). Identity Mappings in Deep Residual Networks. *ECCV 2016.* — Pre-activation residual blocks.

15. **Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T.** (2020). Analyzing and Improving the Image Quality of StyleGAN. *CVPR 2020.* — StyleGAN2: lazy R1 regularization (every 16 minibatches), Tanh output and [-1,1] normalization.

16. **Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M.** (2023). Flow Matching for Generative Modeling. *ICLR 2023.* — Conditional Flow Matching framework used in our CFM pipeline.

17. **Radford, A., Metz, L., & Chintala, S.** (2016). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. *ICLR 2016.* — DCGAN: strided convolutions replacing pooling, Adam with beta_1=0.5 for GAN training.

18. **Brock, A., Donahue, J., & Simonyan, K.** (2019). Large Scale GAN Training for High Fidelity Natural Image Synthesis. *ICLR 2019.* — BigGAN: large-scale GAN with hinge loss.

19. **Sonderby, C. K., Raiko, T., Maaloe, L., Sonderby, S. K., & Winther, O.** (2016). Ladder Variational Autoencoders. *NeurIPS 2016.* — Deterministic warm-up (KL annealing) for training deep VAEs.

20. **Ho, J., Jain, A., & Abbeel, P.** (2020). Denoising Diffusion Probabilistic Models. *NeurIPS 2020.* — DDPM: foundational diffusion model using EMA for inference weights.

21. **Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B.** (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *ICLR 2021.* — Score-based SDE framework with EMA weight averaging.

22. **Zhao, H., Gallo, O., Frosio, I., & Kautz, J.** (2017). Loss Functions for Image Restoration with Neural Networks. *IEEE Transactions on Computational Imaging, 3*(1), 47-57. — Systematic comparison of L1 vs. L2 (MSE) for image restoration; L1 produces sharper results.

23. **Mescheder, L., Geiger, A., & Nowozin, S.** (2018). Which Training Methods for GANs do actually Converge? *ICML 2018.* — R1 gradient penalty for GAN discriminator regularization; provides convergence guarantees.

## 10. Files

| File | Purpose |
|------|---------|
| `models/train_vae_v3.py` | VAE v3 standalone training script with R1 + adaptive weight |
| `models/QuantumLatentCFM_vaev3.py` | Self-contained CFM with embedded VAE v3 |
| `jobs/run_vae_v3.sh` | SLURM script for VAE v3 training (300 epochs, 24h walltime) |
| `jobs/run_cfm_vaev3_quantum.sh` | SLURM script for quantum CFM with VAE v3 |
| `jobs/run_cfm_vaev3_classical.sh` | SLURM script for classical CFM with VAE v3 |
