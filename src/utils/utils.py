import torch

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


def qkclip(G):
    pass


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

def proj_hook(module, input, output):
    
    pass


def max_clip_hook(module, input, output):
    pass


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, slen, head_dim = hidden_states.shape
    num_key_value_heads = 8
    head_dim = 128

    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states.view(batch, slen, num_key_value_heads, -1, head_dim)    
    hidden_states = hidden_states.permute(0, 2, 3, 1, 4) 
    hidden_states = hidden_states[:, :, :, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# === Déclaration du cache global ===
attn_outputs = {}

# === Fonctions de hooks ===
def make_proj_hook(layer_idx, proj_type):
    def hook(module, input, output):
        print(module)
        if layer_idx not in attn_outputs:
            attn_outputs[layer_idx] = {}
        attn_outputs[layer_idx][proj_type] = output
    return hook

def make_self_attn_hook(layer_idx):
    def hook(module, input, output):
        q = attn_outputs.get(layer_idx, {}).get("q")
        k = attn_outputs.get(layer_idx, {}).get("k")
        q = repeat_kv(q,2) # n_rep = 2 = num_attention_heads // num_key_value_heads
        k = repeat_kv(k,2)
        if q is not None and k is not None:
            attn_logits = torch.matmul(q, k.transpose(-1, -2))
            print(f"[Layer {layer_idx}] max(QK^T): {attn_logits.max().item():.4f}")
    return hook


