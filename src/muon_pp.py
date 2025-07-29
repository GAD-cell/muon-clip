import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Tuple
from utils.utils import adam_update, muon_update, max_clip_hook


class MuonPP(Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and all(isinstance(p, tuple) for p in params) and len(params) >= 1 and isinstance(params[0][1], torch.nn.Parameter), "input params should be model.named_parameters()"
        
        #filter Q and K matrix (for clipping) refer to: https://moonshotai.github.io/Kimi-K2/ 
        #filter non 2D-matrix (muon is a 2D matrix optimizer) : https://kellerjordan.github.io/posts/muon/
        muon_group = [] 
        adam_group = []
        
        for name,p in params:
            if len(p.shape) == 2 : 
                muon_group.append(p)
            else : 
                adam_group.append(p) 
        
        
        #if not len(clip_group) : print("Warning: unable to find Q and K projection matrices. No clipping applied.")

        muon_config = dict(lr=0.02, momentum=0.95, weight_decay=0)
        adam_config = dict(lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0)
        
        muon_dic = muon_config.copy()
        muon_dic.update({
            "params": muon_group,
            "use_muon": True
        })

        adam_dic = adam_config.copy()
        adam_dic.update({
            "params": adam_group,
            "use_muon": False
        })
        super().__init__([muon_dic, adam_dic], dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss