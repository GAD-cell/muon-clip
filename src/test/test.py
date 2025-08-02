# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# import torch
# import os
# import sys
# import re
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from muon_clip import MuonClip, MuonConfig

# model_name = "Qwen/Qwen3-0.6B"  
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# config = AutoConfig.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",  
#     torch_dtype=torch.bfloat16,
#     cache_dir="./model_cache"
# )
# muonconfig = MuonConfig()
# optimizer = MuonClip(model, config, muonconfig)

# model.train()
# prompt = "The capital of France is"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# labels = inputs["input_ids"].clone()
# inputs["labels"] = labels


# outputs = model(**inputs)
# loss = outputs.loss

# print(f"Dummy loss: {loss.item()}")

# # Optimizer
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()


# model.eval()



import os
import sys
import torch
import deepspeed
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# === Import custom optimizer ===
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from muon_clip import MuonClip, MuonConfig


# === Load model/tokenizer ===
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)


muon_config = MuonConfig()
optimizer = MuonClip(model, config, muon_config)


config = {
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 1,


  "fp16": {
    "enabled": True,
    "auto_cast": False,
  }
}
# === DeepSpeed Initialization ===
model.train()
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=config
)

# === Dummy batch ===
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}
inputs["labels"] = inputs["input_ids"].clone()

# === Training step ===

outputs = model(**inputs)

optimizer.zero_grad()
loss = outputs.loss
loss.backward()
optimizer.step()

if model.local_rank == 0:
    print(f"[Rank {model.local_rank}] Loss: {loss.item()}")