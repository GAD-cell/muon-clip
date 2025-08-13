import torch
import re
from typing import Tuple,List

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
                #handle_out = module.register_forward_hook(self.make_proj_output_hook(layer_idx, proj_type))
                self.handles.extend([handle_in, handle_out])

            # elif m := re.match(r".*layers\.(\d+)\.self_attn\.(k_proj)$", name):
            #     layer_idx = int(m.group(1))
            #     proj_type = "k_proj"
            #     handle_out = module.register_forward_hook(self.make_proj_output_hook(layer_idx, proj_type))
            #     self.handles.append(handle_out)

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

