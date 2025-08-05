import torch
import re
from typing import Tuple,List

def cans_ortho(X:torch.Tensor, s_interval:Tuple[float,float], num_iterations:int, poly_degrees:List[int]) -> torch.Tensor:
    """
    Apply Chebyshev polynomial approximation to orthogonalize a matrix X.
    
    Args:
        X (torch.Tensor): Input tensor to be orthogonalized.
        s_interval (Tuple[float, float]): Interval for Chebyshev approximation.
        num_iterations (int): Number of iterations for the approximation.
        poly_degrees (List[int]): Degrees of the Chebyshev polynomials to use.
        
    Returns:
        torch.Tensor: Orthogonalized tensor.
    """
    # Placeholder for actual implementation
    return X  # Replace with actual orthogonalization logic


def delta_ortho(X:torch.Tensor, s_interval:Tuple[float,float], poly_degrees:List[int], delta:float, eps:float=1e-7) -> torch.Tensor:

    # Placeholder for actual implementation
    return X  # Replace with actual orthogonalization logic


def remez(s_interval:Tuple[float,float],poly_degree:int ):
    """
    Compute the Remez coefficients for Chebyshev polynomial approximation.
    
    Args:
        s_interval (Tuple[float, float]): Interval for Chebyshev approximation.
        poly_degree (int): Degree of the Chebyshev polynomial.
        
    Returns:
        List[float]: Coefficients of the Chebyshev polynomial.
    """
    # Placeholder for actual implementation
    pass

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
        self.is_registered = False
        self.found_layer = False
        
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

        if self.is_registered : return

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

                self.found_layer = True
        print(f"Hooked {len(self.handles)//3} layers")
        self.is_registered = True

        return 

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        print(f"Removed {len(self.handles)} hooks across {len(self.handles) // 3} layers.")
        self.handles.clear()
        self.is_registered = False
        


hook_recorder = HookRecorder()

import types
from torch.nn import Module 
def override_model(model: Module, hook_recorder):
    original_train = model.train
    original_eval = model.eval

    def new_train(self, mode: bool = True):
        hook_recorder.register_input_hook(self)
        return original_train(mode)

    def new_eval(self):
        hook_recorder.remove_hooks()
        return original_train(False)  # PyTorch eval() is implemented as train(False)

    model.train = types.MethodType(new_train, model)
    model.eval = types.MethodType(new_eval, model)

    return model

