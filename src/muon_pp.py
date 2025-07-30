import torch
from torch.optim import Optimizer
import torch.distributed as dist
from typing import Tuple
from utils.utils import adam_update, muon_update, hook_recorder, repeat_kv
import re



class MuonPP(Optimizer):
    def __init__(self, model, config, lr=0.02, weight_decay=0, momentum=0.95, enable_clipping=True, clipping_threshold=1.0, clipping_alpha=0.5):
        
        #defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        #assert isinstance(params, list) and all(isinstance(p, tuple) for p in params) and len(params) >= 1 and isinstance(params[0][1], torch.nn.Parameter), "Input params should be model.named_parameters()"
        self.enable_clipping = enable_clipping
        found_layer = hook_recorder.register_input_hook(model) # add forward hooks for q_proj and k_proj inputs
        if not found_layer : 
            print("Warning: unable to find q_proj and k_proj layers. No clipping applied.")
            self.enable_clipping = False
        
        self.t = clipping_threshold
        self.alpha = clipping_alpha
        self.model_config = config
        self.n_rep = config.num_attention_heads//config.num_key_value_heads

        #filter Q and K matrix (for clipping) refer to: https://moonshotai.github.io/Kimi-K2/ 
        #filter non 2D-matrix (muon is a 2D matrix optimizer) : https://kellerjordan.github.io/posts/muon/
        muon_group = [] 
        adam_group = []  
         
        for name,p in model.named_parameters():

            m = re.search(r"\d+", name)
            if m and ("q_proj" in name or "k_proj" in name): 
                layer_idx = (int(m.group(0)), "q_proj" if "q_proj" in name else "k_proj")      
            else: layer_idx = None

            if len(p.shape) == 2 : 
                muon_group.append((layer_idx,p))
            else : 
                adam_group.append((layer_idx,p))
        

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
                qk_proj_dic = {}
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
                    if group["param_names"][i] and self.enable_clipping:
                        index = group["param_names"][i][0]
                        if not qk_proj_dic.get(index,None): qk_proj_dic[index] = {}

                        x = hook_recorder.attn_inputs[index]
                        
                        proj = torch.matmul(x, p.transpose(-2, -1))
                        qk_proj_dic[index][group["param_names"][i][1]] = (p,proj) # dic structure : {layer_index: {q_proj : (param,proj), k_proj : (param,proj)}}


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

        #QK-clipping
        for key, value in qk_proj_dic.items(): #iterate over layers
            q_param, q_proj = value["q_proj"]
            k_param, k_proj = value["k_proj"]

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

            max_logits = torch.matmul(q_proj.transpose(-2,-1),k_proj).max() 
            eta = min(self.t/max_logits,1)

            q_param.mul_(torch.pow(eta,self.alpha))
            k_param.mul_(torch.pow(eta,1-self.alpha))

        return loss