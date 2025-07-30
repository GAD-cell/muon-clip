from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from muon_pp import MuonPP

os.environ["HF_TOKEN"] = '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'


model_name = "Qwen/Qwen3-0.6B"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  
    torch_dtype=torch.bfloat16,
    cache_dir="./model_cache"
)

model.train()

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