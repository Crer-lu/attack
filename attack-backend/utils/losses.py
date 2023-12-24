import torch
from torch.nn import CrossEntropyLoss,MSELoss

class DeltaLoss:
    def __init__(self):
        pass

    def __call__(self,real,pred):
        loss,_ = torch.max((real - pred)**2,dim = -1)
        loss = loss.mean()
        return loss

class ReconstructLoss:
    def __init__(self):
        self.image_loss = MSELoss()

    def __call__(self,real,pred):
        loss = self.image_loss(pred,real)
        return loss
    
class ClassificationLoss:
    def __init__(self):
        self.class_loss = CrossEntropyLoss()

    def __call__(self,target,result):
        loss = self.class_loss(result,target)
        return loss
    
class RegulationLoss:
    def __init__(self):
        pass
    def __call__(self,latents):
        loss = torch.clamp_min(torch.abs(latents - 0.5)-0.5,0)**2
        loss = loss.mean()
        return loss
    
class GeneratorLoss:
    def __init__(self):
        pass
    def __call__(self, result_pred):
        loss = -result_pred.mean()
        return loss