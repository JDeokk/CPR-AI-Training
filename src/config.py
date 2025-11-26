import os

class Config:
    # Hardware
    DEVICE = "cuda" if os.path.exists("/proc/driver/nvidia") else "cpu"
    SEED = 10200730
    
    # Data Paths
    DATA_ROOT = "../data"
    CSV_PATH = os.path.join(DATA_ROOT, "csv/yolo_all.csv")
    IMAGE_DIR = os.path.join(DATA_ROOT, "images")
    MASK_DIR = os.path.join(DATA_ROOT, "masks")
    
    # Training Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.0005
    NUM_EPOCHS = 20
    NUM_CLASSES = 128 # EfficientNet feature dimension
    
    # Multi-task Loss Weights
    LAMBDA_H = 0.25  # IHPC
    LAMBDA_D = 0.43  # MCD
    LAMBDA_R = 0.32  # CROC
    
    # Augmentation Settings
    SDXL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
