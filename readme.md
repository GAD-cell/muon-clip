# Muon Optimizer 2.0

This repository presents an implementation of the Muon optimizer, enhanced with the QK-Clipping technique introduced in Kimi K2 and better newton-shulz orthogonalization.

## Key Features

- **QK-Clipping**: Introduces a mechanism to stabilize training by clipping attention logits for each head.
- **Corrected RMS**: Corrected Muon's update RMS to ensure a compatible learning rate between Muon and Adam
- **Esasy to use**: Designed to integrate seamlessly with existing transformer and pytorch architectures. Designed to be used as a regular pytorch optimizer.
- **Scalability**: Optimized for large-scale training scenarios and implemented for DDP training.
- **Efficient orthogonalization**: Designed to improve gradients orthogonalization via CANS method, a better newton-shulz iteration with eigenvalues interval estimation and chebychev polynomials. (**Experimental**)
- **Metrics Logs**: Use W&B or tensorboard to monitor QK-clipping

## How to use

Here's a basic example:

```python
from muon_clip import MuonClip, MuonConfig
from transformers import AutoConfig

# model config can also be a dic with at least num_key_value_heads,num_attention_heads and head_dim keys
model_config = AutoConfig.from_pretrained("{hf_model}")

muon_config = MuonConfig(
    lr: float = 1e-4

    muon_beta: float = 0.95
    muon_decay: float = 0.0
    ns_steps:int = 5 #Number of newton-shulz interations. Increase for more precision during orthogonalization

    adam_betas: Tuple[float, float] = (0.9, 0.95)
    adam_decay: float = 0.0
    adam_eps: float = 1e-10

    enable_clipping: bool = True
    clipping_layers_mapping = {"q_proj":"q_proj","k_proj":"k_proj"} # If using a special model with non standard q_proj and k_proj names. Just change the value to the desired name.
    clipping_threshold: float = 50.0
    clipping_alpha: float = 0.5

    log_max_logits:bool = True
    log_dir: str = "./logs" #leave it empty to disable
    cans_ortho:bool = False # Experimental: Use CANS orthogonalization. Suggest to disable it for now.
    estimate_lower_bound:bool = False 
)

optimizer = MuonClip(model, model_config, muon_config)

model.train() #You must call model.train() after defining the optimizer so that hooks are registered correctly.

```

## Demo
Below a training test with and without clipping.
Notice how the logits are clipped when reaching clipping_threshold.
<img src="./images/max_logits.png" alt="Training max_logits" width="800"/>

## Installation

To install muon-clip just use:

```bash
pip install git+https://github.com/GAD-cell/muon-clip.git@main
```

## Coming soon

-"Zero stage 1" like optimization based on [distributed muon](https://arxiv.org/html/2502.16982v1)\
-Notebooks for training and distributed training with MuonClip
