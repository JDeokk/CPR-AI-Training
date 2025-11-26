# Generative AI Driven CPR Training Improvement

## Overview
This repository contains the implementation of the paper "Generative AI Driven Improvement of Generalized CPR Training Model and Application Study". 
It includes modules for:
1. **Preprocessing**: CPR performer detection using YOLO and Optical Flow analysis (NOC estimation).
2. **Augmentation**: Data augmentation using Stable Diffusion XL Inpainting.
3. **Training**: Multi-task learning (IHPC, MCD, CROC) using EfficientNetV2.

## Structure
- `data/`: Place your CSV and images here.
- `src/`: Source code modules.
- `weights/`: Saved models.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Configure paths in `src/config.py`.
3. Train: `python -m src.train`
4. Evaluate: `python -m src.evaluate`
