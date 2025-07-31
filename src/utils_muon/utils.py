import torch
import re

def newtonschulz(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / (G.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = newtonschulz(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def repeat_kv(hidden_states: torch.Tensor, n_rep: int, num_key_value_heads: int, head_dim: int ) -> torch.Tensor:
    """
    For KV-cache. repeat k_proj to perform q_proj*k_proj.T 
    """
    batch, slen, _ = hidden_states.shape

    if n_rep == 1: #num_attention_head==num_key_value_heads
        return hidden_states.view(batch, num_key_value_heads, slen, head_dim)
    
    hidden_states = hidden_states.view(batch, slen, num_key_value_heads, -1, head_dim)    
    hidden_states = hidden_states.permute(0, 2, 3, 1, 4) 
    hidden_states = hidden_states[:, :, :, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class HookRecorder:
    def __init__(self):
        self.attn_inputs = {}
        self.attn_outputs = {}
        self.handles = []  # Store hook handles 

    def make_proj_input_hook(self, layer_idx, proj_type):
        def hook(module, input, output):
            self.attn_inputs[layer_idx] = input[0]
        return hook

    def make_proj_output_hook(self, layer_idx, proj_type):
        def hook(module, input, output):
            if layer_idx not in self.attn_outputs:
                self.attn_outputs[layer_idx] = {}
            self.attn_outputs[layer_idx][proj_type] = output
        return hook

    def register_input_hook(self, model):
        found_layer = False
        for name, module in model.named_modules():
            if m := re.match(r".*layers\.(\d+)\.self_attn\.(q_proj)$", name):
                layer_idx = int(m.group(1))
                proj_type = "q_proj"
                handle_in = module.register_forward_hook(self.make_proj_input_hook(layer_idx, proj_type))
                handle_out = module.register_forward_hook(self.make_proj_output_hook(layer_idx, proj_type))
                self.handles.extend([handle_in, handle_out])

            elif m := re.match(r".*layers\.(\d+)\.self_attn\.(k_proj)$", name):
                layer_idx = int(m.group(1))
                proj_type = "k_proj"
                handle_out = module.register_forward_hook(self.make_proj_output_hook(layer_idx, proj_type))
                self.handles.append(handle_out)

                found_layer = True
        print(f"Hooked {len(self.handles)//3} layers")
        return found_layer

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        print(f"removed {len(self.handles)} hooks")
        self.handles.clear()
        


hook_recorder = HookRecorder()

import types
from torch.nn import Module 
def override_model(model: Module, hook_recorder):
    def new_train(self):
        hook_recorder.register_input_hook(self)
        original_train()
        return self

    def new_eval(self):
        hook_recorder.remove_hooks()
        original_eval()
        return self

    original_train = model.train
    original_eval = model.eval

    model.train = types.MethodType(new_train, model)
    model.eval = types.MethodType(new_eval, model)

    return model

