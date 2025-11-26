import torch
import torch.nn as nn
import timm
from .config import Config

class MultiTaskModel(nn.Module):
    def __init__(self, nc1=1, nc2=1, nc3=1):
        super(MultiTaskModel, self).__init__()
        self.base_model = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=Config.NUM_CLASSES)
        self.fc_h = nn.Linear(Config.NUM_CLASSES, nc1)
        self.fc_r = nn.Linear(Config.NUM_CLASSES, nc2)
        self.fc_d = nn.Linear(Config.NUM_CLASSES, nc3)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        return self.sig(self.fc_h(x)), self.sig(self.fc_r(x)), self.fc_d(x)

def multi_task_loss(ph, pr, pd, th, tr, td):
    ph, pr, pd = ph.view(-1), pr.view(-1), pd.view(-1)
    loss = (nn.BCELoss()(ph, th) * Config.LAMBDA_H) + (nn.BCELoss()(pr, tr) * Config.LAMBDA_R) + (torch.sqrt(nn.MSELoss()(pd, td)) * Config.LAMBDA_D)
    return loss
