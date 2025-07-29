from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import sys
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from muon_pp import MuonPP


os.environ["HF_TOKEN"] = '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'

model_name = "Qwen/Qwen3-0.6B"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
#config = AutoConfig.from_pretrained(model_name)
#print(config)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  
    torch_dtype=torch.bfloat16,
    cache_dir="./model_cache"
)

model.train()

from utils.utils import make_proj_hook, make_self_attn_hook, attn_outputs


for name, module in model.named_modules():

    if m := re.match(r".*layers\.(\d+)\.self_attn\.(q_proj|k_proj)$", name):
        layer_idx = int(m.group(1))
        proj_type = m.group(2)[0]
        module.register_forward_hook(make_proj_hook(layer_idx, proj_type))

    elif m := re.match(r".*layers\.(\d+)\.self_attn$", name):
        layer_idx = int(m.group(1))
        module.register_forward_hook(make_self_attn_hook(layer_idx))


prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
labels = inputs["input_ids"].clone()
inputs["labels"] = labels

# Passer dans le modèle
outputs = model(**inputs)
loss = outputs.loss

print(f"Dummy loss: {loss.item()}")

# Optimiseur custom
optimizer = MuonPP(list(model.named_parameters()))
optimizer.zero_grad()
loss.backward()
optimizer.step()