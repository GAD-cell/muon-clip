import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Tuple



class MuonPP(Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and all(isinstance(p, tuple) for p in params) and len(params) >= 1 and isinstance(params[0][1], torch.nn.Parameter), "input params should be model.named_parameters()"
        
        #filter Q and K matrix (for clipping) refer to: https://moonshotai.github.io/Kimi-K2/ 
        #filter non 2D-matrix (muon is a 2D parameter optimizer)
        no_clip_group = []
        clip_group = [] # Q and K projections
        adam_group = []
        for name,p in params:
            if "q_proj" in name or "k_proj" in name:
                clip_group.append(p)
            elif len(p.shape) == 2 : 
                no_clip_group.append(p)
            else : 
                adam_group.append(p)
        
        if not len(clip_group) : print("Warning: unable to find Q and K projection matrices. No clipping applied.")


        
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            params = group["params"]
            
        pass