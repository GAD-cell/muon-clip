import torch
from torch.optim import Optimizer
import torch.distributed as dist

import re
import wandb
from typing import Tuple
from dataclasses import dataclass

from utils_muon import adam_update, muon_update, hook_recorder, repeat_kv


@dataclass
class MuonConfig:
    muon_lr: float = 5e-4
    muon_momentum: float = 0.95
    muon_decay: float = 0.0
    
    enable_clipping: bool = True
    clipping_threshold: float = 20.0
    clipping_alpha: float = 0.5

    adam_lr: float = 5e-4
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_decay: float = 0.0
    adam_eps: float = 1e-10



# Base muon implementation is from https://github.com/KellerJordan/Muon
class MuonClip(Optimizer):
    '''
    Hybrid optimizer that combines Muon and Adam optimization strategies with optional input clipping 
    for specific attention projection layers (q_proj and k_proj).

    This optimizer separates model parameters into two groups:
    - 2D parameters and q_proj and k_proj layers (optimized using Muon)
    - All other parameters (optimized using Adam)

    If enabled, an input clipping mechanism is applied to the inputs of q_proj and k_proj layers 
    based on a configurable threshold.

    Args:
        model (nn.Module): The model whose parameters are to be optimized.
        model_config (Any): Configuration object containing model architecture details. Must include:
            - num_attention_heads (int)
            - num_key_value_heads (int)
            - head_dim (int)
        muon_config (MuonConfig): Configuration dataclass containing all optimizer and clipping parameters.

    Notes:
        - Clipping hooks are only applied if q_proj and k_proj layers are found and `enable_clipping` is True.
        - Only 2D parameters (i.e., weight matrices) are optimized with Muon.
        - Parameter groups are constructed manually with the `use_muon` flag for later use in the step function.

    References:
        - Muon optimizer: https://kellerjordan.github.io/posts/muon/
        - Original implementation: https://github.com/KellerJordan/Muon
        - Kimi-K2 filtering (clipping idea): https://moonshotai.github.io/Kimi-K2/
    '''

    def __init__(self, model, model_config, muon_config: MuonConfig):
        self.enable_clipping = muon_config.enable_clipping
        found_layer = hook_recorder.register_input_hook(model)  # q_proj / k_proj forward hooks
        if not found_layer:
            print("Warning: unable to find q_proj and k_proj layers. No clipping applied.")
            self.enable_clipping = False

        self.t = muon_config.clipping_threshold
        self.alpha = muon_config.clipping_alpha
        self.model_config = model_config
        self.muon_config = muon_config
        self.n_rep = model_config.num_attention_heads // model_config.num_key_value_heads
        self._step = 0

        muon_group = []
        adam_group = []

        for name, p in model.named_parameters():
            m = re.search(r"\d+", name) #add layer id for clipping later
            layer_idx = (int(m.group(0)), "q_proj" if "q_proj" in name else "k_proj") if m and ("q_proj" in name or "k_proj" in name) else None

            if p.ndim == 2:
                muon_group.append((layer_idx, p))
            else:
                adam_group.append((layer_idx, p))

        muon_dic = {
            "params": muon_group,
            "lr": muon_config.muon_lr,
            "momentum": muon_config.muon_momentum,
            "weight_decay": muon_config.muon_decay,
            "use_muon": True
        }

        adam_dic = {
            "params": adam_group,
            "lr": muon_config.adam_lr,
            "betas": muon_config.adam_betas,
            "eps": muon_config.adam_eps,
            "weight_decay": muon_config.adam_decay,
            "use_muon": False
        }

        super().__init__([muon_dic, adam_dic], {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                qk_proj_dic = {}
                old_proj_dic = {}
                for i,p in enumerate(group["params"]):
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                
                    # save proj 
                    if group["param_names"][i] :
                        index = group["param_names"][i][0]
                        proj_type = group["param_names"][i][1]
                        if not old_proj_dic.get(index,None): old_proj_dic[index] = {}
                        output = hook_recorder.attn_outputs[index][proj_type]
                        old_proj_dic[index][proj_type] = {'out':output}

                        if self.enable_clipping:
                            if not qk_proj_dic.get(index,None): qk_proj_dic[index] = {}
                            x = hook_recorder.attn_inputs[index]
                            proj = torch.matmul(x, p.transpose(-2, -1))
                            qk_proj_dic[index][proj_type] = {'param':p, 'proj':proj} 


            else: #Adam
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


        #max logits log
        global_max = 0.0
        for key, value in old_proj_dic.items():
            q_out = value["q_proj"]["out"]
            k_out = value["k_proj"]["out"]
            old_attn_logits = torch.matmul(q_out,k_out.transpose(-2,-1))
            per_head_max = old_attn_logits.amax(dim=(-2, -1)).amax(dim=0) 
            old_local_max = per_head_max.amax(dim=0).item()
            global_max = old_local_max if old_local_max > global_max else global_max
        
        wandb.log({"max_logits": global_max}, step=self._step)

        #QK-clipping
        for key, value in qk_proj_dic.items(): #iterate over layers
            q_param, q_proj = value["q_proj"]["param"], value["q_proj"]["proj"] 
            k_param, k_proj = value["k_proj"]["param"], value["k_proj"]["proj"] 

            q_proj = repeat_kv(
                                q_proj,
                                self.n_rep,
                                self.model_config.num_key_value_heads,
                                self.model_config.head_dim)
            k_proj = repeat_kv(
                                k_proj,
                                self.n_rep,
                                self.model_config.num_key_value_heads,
                                self.model_config.head_dim)

            attn_logits = torch.matmul(q_proj,k_proj.transpose(-2,-1))
            per_head_max = attn_logits.amax(dim=(-2, -1)).amax(dim=0) # 1 max per head 
            per_head_eta = (self.t / per_head_max).clamp(max=1.0)
            per_head_eta = per_head_eta.unsqueeze(0).unsqueeze(-1)
            
            #separate query params heads and scale by eta per head
            q = q_param.data.transpose(-2,-1) #if transpose else q_param.data
            q = q.data.view(q.size(0), -1, self.model_config.head_dim) # [in_dim,out_dim] -> [in_dim,num_head,head_dim]
            q *= per_head_eta**self.alpha
            q = q.view(q.size(0),-1) # original size (in_dim,out_dim)
            q = q.transpose(-2,-1)
            q_param.data.copy_(q.clone())
            
            #separate key params heads, scale eta per head and take into account kv cache
            #For handling key heads, we take the minimum eta value within each KV head group, 
            #applying the strongest (smallest) rescaling factor to ensure stability in the worst-case scenario.    
            k = k_param.data.transpose(-2,-1) 
            k = k.data.view(k.size(0), -1, self.model_config.head_dim)
            per_key_head_eta = per_head_eta.view(per_head_eta.size(0), k.size(1), -1).min(dim=2).values #notice min for each group
            per_key_head_eta = per_key_head_eta.unsqueeze(-1)
            k *= per_key_head_eta**(1-self.alpha)
            k = k.view(k.size(0),-1) # original size (in_dim,out_dim)
            k = k.transpose(-2,-1)
            k_param.data.copy_(k.clone())        

        self._step += 1

        return loss