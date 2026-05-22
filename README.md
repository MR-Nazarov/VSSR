# VSSR: View-Specialized Sequential Refinement for Accelerated MRI

**Post-Reconstruction Volumetric Refinement for Accelerated MRI via Cross-Plane Consistency**

*IEEE Engineering in Medicine and Biology Conference (EMBC) 2026 — Paper ID: 4868*

---

## Overview

VSSR is a post-reconstruction refinement framework for accelerated MRI. Rather than treating volumetric reconstruction as a single-model problem, VSSR decomposes it into a cascade of three view-specialized expert networks — each dedicated to one anatomical plane (sagittal, axial, coronal) — that refine the volume sequentially. Each expert corrects the residual artifacts left by the previous stage, exploiting the complementary information available along orthogonal planes.

<p align="center">
  <img src="assets/method_overview.png" width="800" alt="VSSR pipeline overview"/>
</p>

### Key Contributions

- **View-specialized experts:** Three 2D networks, each trained on slices from a dedicated anatomical plane, target different spatial frequency artifacts introduced by k-space undersampling.
- **Sequential refinement with cross-plane consistency:** Each expert takes the output of the previous stage as input and is supervised against the fully-sampled ground truth, enforcing consistency across planes throughout training.
- **Multi-component loss:** A weighted combination of L1, SSIM, and perceptual (VGG) losses that jointly optimizes pixel fidelity, structural similarity, and perceptual quality.
- **Architecture-agnostic:** The pipeline supports CNN backbones (RDUNet [[1]](#references), DnCNN [[2]](#references), 3D U-Net [[3]](#references)) and Swin Transformer (SUNet [[4]](#references)) interchangeably.

---

## Method

The reconstruction pipeline operates in three stages:

```
Undersampled MRI
      │
      ▼
 [Expert 1 – Sagittal]   ← trained on sagittal slices
      │
      ▼
 [Expert 2 – Axial]      ← trained on axial slices of Expert 1 output
      │
      ▼
 [Expert 3 – Coronal]    ← trained on coronal slices of Expert 2 output
      │
      ▼
 Refined 3D Volume
```

**Sequential training** ensures each expert is trained on the actual distribution it will encounter at test time, eliminating the covariate shift that degrades independently trained cascades.

---

## Requirements

```bash
torch >= 2.0.0
monai >= 1.3.0
hydra-core >= 1.3.0
torchmetrics >= 1.0.0
nibabel
pydicom
wandb
timm
einops
opencv-python
scikit-image
scipy
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

The pipeline is evaluated on the **IXI dataset** [[5]](#references) (8× undersampling) and a prospectively accelerated dataset acquired at **Sheba Medical Center**.

Data splits are specified via JSON files. Expected structure:

```
data/
└── ixi/
    ├── train.json
    ├── val.json
    └── test.json
```

Each JSON entry contains paths to the undersampled input and fully-sampled ground truth NIfTI volumes (`.nii.gz`).

To generate intermediate data files needed for sequential training stages 2 and 3:

```bash
python generate_intermediate_data.py --experiment ASC_8x --stage 1
```

---

## Pre-trained Weights

Pre-trained models are hosted on HuggingFace:

> **[Lexer1/VSSR on HuggingFace](https://huggingface.co/Lexer1/VSSR)**

Download and place checkpoints under `models/`:

```
models/
├── vssr_stage1_sagittal.pth
├── vssr_stage2_axial.pth
└── vssr_stage3_coronal.pth
```

---

## Training

All experiments are configured with [Hydra](https://hydra.cc/). To train the full VSSR pipeline (ASC ordering: Axial → Sagittal → Coronal):

**Stage 1 – Axial expert:**
```bash
python main.py experiment=ASC_8x/1_axial
```

**Stage 2 – Sagittal expert (takes Stage 1 output as input):**
```bash
python main.py experiment=ASC_8x/2_sagittal_from_axial
```

**Stage 3 – Coronal expert (takes Stage 2 output as input):**
```bash
python main.py experiment=ASC_8x/3_coronal_from_sagittal
```

Override any parameter directly from the command line:

```bash
python main.py experiment=ASC_8x/1_axial model=sunet training.batch_size=8
```

Experiment tracking is handled via [Weights & Biases](https://wandb.ai/). Set `wandb.project` in `conf/config.yaml` or pass it as a CLI override.

---

## Inference

To run inference with a trained cascade on a test set:

```bash
python main.py experiment=ASC_8x/3_coronal_from_sagittal mode=test
```

For sliding-window inference on large volumes (configurable overlap and ROI size):

```bash
python main.py experiment=ASC_8x/3_coronal_from_sagittal mode=test inference.sliding_window=true
```

Outputs are saved as `.nii.gz` files with the original affine and header metadata preserved.

---

## Results

| Method | PSNR (dB) | SSIM |
|---|---|---|
| Zero-filled (8×) | — | — |
| Single expert – Sagittal | — | — |
| Single expert – Axial | — | — |
| Single expert – Coronal | — | — |
| VSSR (Ours) | — | — |

*Results on IXI test set, 8× acceleration. Full results in the paper.*

---

## Repository Structure

```
VSSR/
├── conf/                    # Hydra configuration files
│   ├── config.yaml
│   ├── model/               # Architecture configs (sunet, dncnn, rdunet, unet3D)
│   ├── loss/                # Loss function configs
│   └── experiments/         # Per-experiment overrides (added after supervisor approval)
├── src/
│   └── networks/            # Model definitions
│       ├── SUNet.py         # Swin Transformer U-Net [4]
│       ├── DnCNN.py         # Denoising CNN [2]
│       ├── DnCNN3D.py
│       ├── RDUNet.py        # Residual Dense U-Net [1]
│       └── Unet3D.py        # 3D U-Net [3]
├── datasets/                # Dataset classes (MONAI-based)
├── data_utils/              # Scripts to generate sequential training data
├── losses.py                # CombinedLoss (L1 + SSIM + Perceptual)
├── metrics.py               # PSNR / SSIM wrappers
├── trainer.py               # Training engine
├── main.py                  # Entry point
└── requirements.txt
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{nazarov2026vssr,
  title     = {Post-Reconstruction Volumetric Refinement for Accelerated {MRI} via Cross-Plane Consistency},
  author    = {Nazarov, Alexander and Kiryati, Nahum and Roizen, Dani and Greenberg, Gahl and Mayer, Arnaldo},
  booktitle = {2026 IEEE Engineering in Medicine and Biology Conference (EMBC)},
  year      = {2026},
  note      = {Paper ID: 4868}
}
```

---

## References

The backbone architectures used in this work are based on the following papers. Please cite them if you use the respective models.

**[5]** Biomedical Image Analysis Group, Imperial College London. "IXI Dataset." *Information eXtraction from Images* (EPSRC GR/S21533/02). https://brain-development.org/ixi-dataset/
```bibtex
@misc{ixi_dataset,
  title        = {{IXI} Dataset},
  author       = {{Biomedical Image Analysis Group, Imperial College London}},
  howpublished = {\url{https://brain-development.org/ixi-dataset/}},
  note         = {Information eXtraction from Images (EPSRC GR/S21533/02). Licensed under CC BY-SA 3.0},
  year         = {2007}
}
```

**[1]** Gurrola-Ramos, J., Dalmau, O., and Alarcón, T. E. "A Residual Dense U-Net Neural Network for Image Denoising." *IEEE Access*, vol. 9, pp. 31742–31754, 2021.
```bibtex
@article{gurrola2021residual,
  title     = {A Residual Dense {U-Net} Neural Network for Image Denoising},
  author    = {Gurrola-Ramos, Javier and Dalmau, Oscar and Alarc{\'o}n, Teresa E.},
  journal   = {IEEE Access},
  volume    = {9},
  pages     = {31742--31754},
  year      = {2021},
  publisher = {IEEE},
  doi       = {10.1109/ACCESS.2021.3061062}
}
```

**[2]** Zhang, K., Zuo, W., Chen, Y., Meng, D., and Zhang, L. "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising." *IEEE Transactions on Image Processing*, vol. 26, no. 7, pp. 3142–3155, 2017.
```bibtex
@article{zhang2017beyond,
  title     = {Beyond a {Gaussian} Denoiser: Residual Learning of Deep {CNN} for Image Denoising},
  author    = {Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal   = {IEEE Transactions on Image Processing},
  volume    = {26},
  number    = {7},
  pages     = {3142--3155},
  year      = {2017},
  publisher = {IEEE},
  doi       = {10.1109/TIP.2017.2662206}
}
```

**[3]** Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., and Ronneberger, O. "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." *MICCAI*, 2016.
```bibtex
@inproceedings{cicek20163d,
  title     = {{3D U-Net}: Learning Dense Volumetric Segmentation from Sparse Annotation},
  author    = {{\c{C}}i{\c{c}}ek, {\"O}zg{\"u}n and Abdulkadir, Ahmed and Lienkamp, Soeren S. and Brox, Thomas and Ronneberger, Olaf},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2016},
  series    = {Lecture Notes in Computer Science},
  volume    = {9901},
  pages     = {424--432},
  year      = {2016},
  publisher = {Springer International Publishing},
  doi       = {10.1007/978-3-319-46723-8\_49}
}
```

**[4]** Fan, C.-M., Liu, T.-J., and Liu, K.-H. "SUNet: Swin Transformer UNet for Image Denoising." *IEEE ISCAS*, 2022.
```bibtex
@inproceedings{fan2022sunet,
  title     = {{SUNet}: Swin Transformer {UNet} for Image Denoising},
  author    = {Fan, Chi-Mao and Liu, Tsung-Jung and Liu, Kuan-Hsien},
  booktitle = {2022 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages     = {2333--2337},
  year      = {2022},
  publisher = {IEEE},
  doi       = {10.1109/ISCAS48785.2022.9937486}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
