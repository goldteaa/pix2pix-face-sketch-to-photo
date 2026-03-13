# Pix2Pix GAN – Face Sketch to Photo Synthesis

## Project Overview

This project implements a **Pix2Pix Conditional Generative Adversarial Network (cGAN)** to translate **hand-drawn face sketches into realistic face photographs**.

The model learns a mapping between sketches and corresponding real images using supervised image-to-image translation techniques.

Sketch-to-photo synthesis is an important computer vision task with applications in **law enforcement, forensic investigations, and criminal identification systems**.

---

## Objective

The goal of this project is to train a deep learning model capable of reconstructing realistic face photographs from hand-drawn sketches.

By learning the relationship between sketch images and real photographs, the model demonstrates how **Generative Adversarial Networks (GANs)** can be used for image translation tasks.

---

## Dataset

The model was trained using the **CUHK Face Sketch Database (CUFS)**.

Dataset characteristics:

* 188 paired sketch–photo images
* Each sketch corresponds to a real face photograph
* Frontal face images
* Controlled lighting conditions

Data split:

* 80% training
* 20% validation

Images were resized to **256 × 256 pixels** for training.

---

## Tools & Technologies

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Generative Adversarial Networks (GANs)
* Computer Vision

---

## Methodology

The project implements the **Pix2Pix architecture**, which consists of two neural networks trained together.

### Generator

The generator converts a **sketch → photo**.

Architecture features:

* U-Net encoder–decoder structure
* Skip connections to preserve spatial information
* Learns to generate realistic face images from sketches

### Discriminator

The discriminator evaluates whether an image is:

* a real photograph
* or a generated image

Architecture features:

* PatchGAN discriminator
* Evaluates local image patches
* Encourages realistic texture generation

Training details:

* Optimizer: Adam
* Learning rate: 0.0002
* Beta1: 0.5
* Training epochs: 100

Loss functions used:

* Adversarial loss
* L1 reconstruction loss

---

## Results

During training the model gradually improved the quality of generated images.

Early training stages produced **blurry outputs with limited facial detail**, while later epochs generated more realistic face structures and improved hair and facial features.

The **best-performing generator model was obtained at epoch 69**.

---

## Training Loss Curve

![Loss Curve](loss_curves.png)

The generator loss decreases during training as the model learns to produce more realistic face images.

---

## Example Generated Images

### Best Generated Image

![Best Preview](outputs_pix2pix/best_preview.png)

### Training Progression

| Epoch 1                            | Epoch 50                           | Epoch 100                          |
| ---------------------------------- | ---------------------------------- | ---------------------------------- |
| ![](outputs_pix2pix/epoch_001.png) | ![](outputs_pix2pix/epoch_050.png) | ![](outputs_pix2pix/epoch_100.png) |

These examples illustrate how the generated faces improve during training.

---

## Project Structure

```id="pix2pix_structure"
pix2pix-face-sketch-to-photo
│
├── notebook
│   └── CUFS_Pix2Pix.ipynb
│
├── models
│   ├── pix2pix_generator_best.keras
│   └── pix2pix_generator_cufs.keras
│
├── outputs_pix2pix
│   generated images during training
│
├── project_report.pdf
└── README.md
```

---

## How to Run

1. Clone the repository

```id="pix2pix_clone"
git clone https://github.com/goldteaa/pix2pix-face-sketch-to-photo.git
```

2. Install dependencies

```id="pix2pix_install"
pip install tensorflow numpy matplotlib
```

3. Open the notebook and run the training process

```
CUFS_Pix2Pix.ipynb
```

---

## Future Improvements

* Train the model on larger datasets
* Improve image resolution with advanced GAN architectures
* Experiment with CycleGAN for unpaired image translation
* Add perceptual loss functions to improve visual quality
