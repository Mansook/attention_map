import torch
import torch.nn as nn
import torch.nn.functional as F

class SHAP(nn.Module):
    def __init__(self,model,config_dict):
        super(SHAP,self).__init__()
        self.model = model.eval()
        self.input_size = config_dict.get('input_size',(96,96))
        self.baseline = config_dict.get('baseline',0)
        self.feature_maps = None
        self.gradients = None