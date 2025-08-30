from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import os
import sys
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from muon import MuonClip, MuonConfig

model_name = "Qwen/Qwen3-0.6B"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  
    torch_dtype=torch.bfloat16,
    cache_dir="/Volumes/Mac_ext/code_projects/model_cache"
)
muonconfig = MuonConfig()
optimizer = MuonClip(model, config, muonconfig)

model.train()
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
labels = inputs["input_ids"].clone()
inputs["labels"] = labels


outputs = model(**inputs)
loss = outputs.loss

print(f"Dummy loss: {loss.item()}")

# Optimizer
optimizer.zero_grad()
loss.backward()
optimizer.step()


model.eval()