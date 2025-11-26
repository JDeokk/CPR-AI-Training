import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import f1_score
from .config import Config
from .utils import set_seed
from .dataset import create_dataloaders
from .model import MultiTaskModel, multi_task_loss

def train():
    set_seed(Config.SEED)
    train_dl, val_dl, _ = create_dataloaders()
    
    model = MultiTaskModel().to(Config.DEVICE)
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
    
    opt = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3)
    
    best_loss = np.Inf
    for ep in range(Config.NUM_EPOCHS):
        model.train()
        tr_loss = 0.0
        for d, th, tr, td in tqdm(train_dl, desc=f"Ep {ep+1}"):
            d, th, tr, td = d.to(Config.DEVICE), th.to(Config.DEVICE).float(), tr.to(Config.DEVICE).float(), td.to(Config.DEVICE).float()
            opt.zero_grad()
            ph, pr, pd = model(d)
            loss = multi_task_loss(ph, pr, pd, th, tr, td)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        preds, targs = [], []
        with torch.no_grad():
            for d, th, tr, td in val_dl:
                d, th, tr, td = d.to(Config.DEVICE), th.to(Config.DEVICE).float(), tr.to(Config.DEVICE).float(), td.to(Config.DEVICE).float()
                ph, pr, pd = model(d)
                val_loss += multi_task_loss(ph, pr, pd, th, tr, td).item()
                preds.extend(ph.cpu().numpy()); targs.extend(th.cpu().numpy())

        val_loss /= len(val_dl)
        sch.step(val_loss)
        f1 = f1_score(targs, np.round(preds), average='weighted')
        
        print(f"Ep {ep+1} | Train: {tr_loss/len(train_dl):.4f} | Val: {val_loss:.4f} | Hand F1: {f1:.4f}")
        
        if val_loss <= best_loss:
            if not os.path.exists("weights"): os.makedirs("weights")
            torch.save(model.state_dict(), 'weights/best_model.pt')
            best_loss = val_loss

if __name__ == "__main__":
    train()
