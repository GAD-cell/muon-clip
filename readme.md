# Muon Optimizer with QK-Clipping

This repository presents an implementation of the Muon optimizer, enhanced with the QK-Clipping technique introduced in Kimi K2. Muon is a optimizer designed for efficient training of deep neural networks, particularly in large-scale settings like for LLM training. The integration of QK-Clipping enhances training stability by addressing issues related to attention score explosions in transformer models.

## Key Features

- **QK-Clipping**: Introduces a mechanism to stabilize training by clipping attention logits.
- **Compatibility**: Designed to integrate seamlessly with existing transformer architectures.
- **Scalability**: Optimized for large-scale training scenarios.

## QK-Clipping Explained

QK-Clipping is a technique that addresses the issue of exploding attention logits in transformer models. By rescaling the query and key matrices during training, QK-Clipping ensures that the attention scores remain within a stable range, preventing instability and promoting smoother convergence. This method was instrumental in the pre-training of Kimi K2 on 15.5 trillion tokens without any loss spikes.

For more details, refer to the following resources:

- [Kimi K2: Open Agentic Intelligence (arXiv)](https://arxiv.org/abs/2507.20534)

## Installation

To install the Muon optimizer with QK-Clipping just use:

```bash
pip install git+https://github.com/GAD-cell/muon-clip.git@main
```