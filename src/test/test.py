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

optimizer = MuonPP(list(model.named_parameters()))
optimizer.zero_grad()
optimizer.step()

# prompt = "Give me a short introduction to large language model."

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# with torch.no_grad():
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=50,
#         do_sample=True,
#         top_p=0.95,
#         temperature=0.7
#     )

# # Résultat
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))