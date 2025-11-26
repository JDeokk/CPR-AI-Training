import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from .config import Config
from .dataset import create_dataloaders
from .model import MultiTaskModel, multi_task_loss

def evaluate():
    _, _, test_dl = create_dataloaders()
    model = MultiTaskModel().to(Config.DEVICE)
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('weights/best_model.pt'))
    model.eval()
    
    ph_l, pr_l, pd_l = [], [], []
    th_l, tr_l, td_l = [], [], []
    loss = 0.0
    
    with torch.no_grad():
        for d, th, tr, td in tqdm(test_dl, desc="Testing"):
            d, th, tr, td = d.to(Config.DEVICE), th.to(Config.DEVICE).float(), tr.to(Config.DEVICE).float(), td.to(Config.DEVICE).float()
            ph, pr, pd = model(d)
            loss += multi_task_loss(ph, pr, pd, th, tr, td).item()
            
            ph_l.extend(ph.cpu().numpy()); th_l.extend(th.cpu().numpy())
            pr_l.extend(pr.cpu().numpy()); tr_l.extend(tr.cpu().numpy())
            pd_denorm = (pd.view(-1).cpu().numpy() * 43) + 20
            td_denorm = (td.view(-1).cpu().numpy() * 43) + 20
            pd_l.extend(pd_denorm)
            td_l.extend(td_denorm)
            
    print(f"Test Loss: {loss/len(test_dl):.4f}")
    print(f"Hand F1: {f1_score(th_l, np.round(ph_l), average='weighted'):.4f} | AUC: {roc_auc_score(th_l, ph_l):.4f}")
    print(f"Rel F1: {f1_score(tr_l, np.round(pr_l), average='weighted'):.4f} | AUC: {roc_auc_score(tr_l, pr_l):.4f}")
    print(f"Depth MAE: {np.mean(np.abs(np.array(td_l)-np.array(pd_l))):.4f} mm")

if __name__ == "__main__":
    evaluate()
