# Generative AI Driven CPR Training Improvement

<img width="550" height="400" alt="CPR_framework" src="https://github.com/user-attachments/assets/51041a40-079f-47ac-be1f-15bcbd3ba83c" />

## 📖 Overview
This repository contains the official implementation of the paper **"Generative AI Driven Improvement of Generalized CPR Training Model and Application Study"**.

The proposed framework utilizes a sensorless approach to evaluate CPR quality using smartphone cameras. It consists of three main modules:
1.  **Preprocessing**: CPR performer detection using **YOLO** and **Optical Flow** analysis for compression count (NOC) estimation.
2.  **Augmentation**: Data augmentation using **Stable Diffusion XL Inpainting** to simulate diverse environments.
3.  **Training**: Multi-task learning (IHPC, MCD, CROC) using **EfficientNetV2**.

## 💾 Model Checkpoints
You can download the pre-trained model weights (`.pt` files) from the link below:

👉 **[Download Model Weights (Google Drive)](https://drive.google.com/drive/folders/1LS-kU9rb3Ol_9F06IULehSbnt1M9ObeH?usp=drive_link)**

> **Note:** After downloading, please place the weight file (e.g., `best_model.pt` or `yolo_all_0404_lr_8e-4_6.pt`) into the `weights/` directory.

## 📂 Project Structure
```bash
CPR-AI-Training/
├── data/                   # Place your dataset here
│   ├── csv/                # CSV annotations (e.g., yolo_all.csv)
│   ├── images/             # Image files
│   └── masks/              # Mask files for inpainting
├── src/                    # Source code modules
│   ├── config.py           # Configuration (Paths, Hyperparameters)
│   ├── dataset.py          # Data loading & augmentation
│   ├── model.py            # Multi-task EfficientNet model
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── ...
├── weights/                # Place downloaded .pt files here
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Configure paths in `src/config.py`.
3. Train: `python -m src.train`
4. Evaluate: `python -m src.evaluate`
