# Muon Optimizer with QK-Clipping

This repository presents an implementation of the Muon optimizer, enhanced with the QK-Clipping technique introduced in Kimi K2.

## Key Features

- **QK-Clipping**: Introduces a mechanism to stabilize training by clipping attention logits.
- **Esasy to use**: Designed to integrate seamlessly with existing transformer and pytorch architectures.
- **Scalability**: Optimized for large-scale training scenarios.

## QK-Clipping Explained

QK-Clipping is a technique that addresses the issue of exploding attention logits in transformer models. By rescaling the query and key matrices during training, QK-Clipping ensures that the attention scores remain within a stable range, preventing instability and promoting smoother convergence. This method was instrumental in the pre-training of Kimi K2 on 15.5 trillion tokens without any loss spikes.
The clipping is applied per heads.

For more details, refer to the following resources:

- [Kimi K2: Open Agentic Intelligence (arXiv)](https://arxiv.org/abs/2507.20534)

## How to use

Here's a basic example of how to use the MuonClip optimizer:

```python

```
## Installation

To install the Muon optimizer with QK-Clipping just use:

```bash
pip install git+https://github.com/GAD-cell/muon-clip.git@main
```

## Future implementation

-Currently working on an improved version of newton-shulz orthogonalization based on [Accelerating Newton-Shulz Iteration](https://arxiv.org/pdf/2506.10935v1) paper
-Multi-gpu support