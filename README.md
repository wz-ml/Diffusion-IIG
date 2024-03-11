# Diffusion-IIG: Diffusion Implicit Image Generation

![Diffusion-IIG logo](OIG4.jpg)

A reimplemention of the DDIM sampling / generation algorithm, as a precursor to implicit image generation via invertible DDIMs.
We compare three different modes of the latest generative models, provide a library of options for noise schedulers / samplers, and analyze the best hyperparameters and model architecture decisions for generative models of various scales.

## Background:
Diffusion models have emerged as a powerful class of generative models, capable of producing high-quality, diverse samples across a range of domains, including images, audio, and text. The Diffusion-IIG project specifically focuses on image generation, leveraging the invertibility and consistency of DDIM models to generate any kind of implicit image - depthmaps, segmentation maps, or albedo maps - given an arbitrary input image.

Our main goal right now is to provide a comprehensive codebase that allows easy experimentation, modification, and extension of diffusion + pixelCNN models. Stay tuned for our implicit image generation work!

## Installation:
To get started with Diffusion-IIG, you'll need to have Python 3.6+ and PyTorch 1.0+ installed. You can then clone this repository and follow the instructions below to setup the necessary packages.
1. `git clone Diffusion-IIG`
2. Run the following:
```bash
conda create Diffusion_IIG python=3.9.3 ipython
conda activate Diffusion_IIG
pip install -r requirements.txt
```


## Todo:
- [] Implement PixelCNN benchmark.
- [] Implement DDPM benchmark.
    - [x] Noise scheduler library.
    - [] Sampler library.
- [] Implement DDIM benchmark.
    - [x] UNet architecture.
    - [] Training loop.
