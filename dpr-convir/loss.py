import torch
import torch.nn as nn

def InBatchKLLoss(logits, ):
    pass

def InBatchNegativeCELoss(logits):
    CELoss = nn.CrossEntropyLoss()
    labels = torch.arange(0, logits.size(0), device=logits.device)
    return CELoss(logits, labels)

def PairwiseCELoss(scores):
    CELoss = nn.CrossEntropyLoss()
    logits = scores.view(2, -1).permute(1, 0) # (B*2 1) -> (B 2)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return CELoss(logits, labels)
