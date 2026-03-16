import torch
import torch.nn as nn



class ClsLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ClsLoss, self).__init__()
        weight = kwargs.get('weight', None)
        self.loss_func = nn.CrossEntropyLoss(reduction="mean", weight= weight)

    def forward(self, predicts, labels):
        loss = self.loss_func(predicts, labels)
        return {"loss": loss}