import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from .config import Config

def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose([
            A.Normalize(), A.HorizontalFlip(p=0.5), A.Rotate(limit=10, p=0.5),
            A.OneOf([A.MotionBlur(p=0.2), A.GaussianBlur(p=0.2)], p=0.2),
            A.RandomBrightnessContrast(p=0.3), A.Sharpen(p=0.3),
            A.PixelDropout(p=0.4, dropout_prob=0.03), ToTensorV2(),
        ])
    return A.Compose([A.Normalize(), ToTensorV2()])

class CPRDataset(Dataset):
    def __init__(self, img_paths, hand=None, release=None, depth=None, transforms=None):
        self.img_paths = img_paths
        self.hand = hand
        self.release = release
        self.depth = depth
        self.transforms = transforms

    def __getitem__(self, idx):
        x = np.array(Image.open(self.img_paths[idx]).convert('RGB'))

        y_h = np.array(self.hand[idx]) if self.hand is not None else 0
        y_r = np.array(self.release[idx]) if self.release is not None else 0
        y_d = (np.array(self.depth[idx]) - 20) / 43 if self.depth is not None else 0

        if self.transforms: x = self.transforms(image=x)['image']
        return x, y_h, y_r, y_d
        
    def __len__(self): return len(self.img_paths)

def create_dataloaders():
    df = pd.read_csv(Config.CSV_PATH)
    train_df = df.iloc[:15424].reset_index(drop=True)
    val_df = df.iloc[15424:18728].reset_index(drop=True)
    test_df = df.iloc[18728:].reset_index(drop=True)
    
    le = LabelEncoder()
    if "P_Hand_er_C" in df.columns:
        for d in [train_df, val_df, test_df]:
            d["P_Hand_er_C"] = le.fit_transform(d["P_Hand_er_C"].astype(str))
            d["P_Imcomp_release_C"] = le.fit_transform(d["P_Imcomp_release_C"].astype(str))

    def get_data(d):
        return d['critical_based_path'].values, d['P_Hand_er_C'].values, d["P_Imcomp_release_C"], d['P_Depth'].values

    tr_p, tr_h, tr_r, tr_d = get_data(train_df)
    val_p, val_h, val_r, val_d = get_data(val_df)
    te_p, te_h, te_r, te_d = get_data(test_df)

    tr_ds = CPRDataset(tr_p, tr_h, tr_r, tr_d, get_transforms("train"))
    val_ds = CPRDataset(val_p, val_h, val_r, val_d, get_transforms("valid"))
    te_ds = CPRDataset(te_p, te_h, te_r, te_d, get_transforms("valid"))

    return (DataLoader(tr_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4),
            DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4),
            DataLoader(te_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4))
