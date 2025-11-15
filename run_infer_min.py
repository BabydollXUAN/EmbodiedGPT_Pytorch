import os, sys, torch
sys.path.append(os.path.dirname(__file__))
from transformers import AutoTokenizer, AutoModelForCausalLM

ckpt_dir = "./checkpoints/Embodied_family_7btiny"   # 你下载/解压的 tiny 权重目录
assert os.path.exists(ckpt_dir), f"missing: {ckpt_dir}"
tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
m = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=dtype, device_map="auto")
prompt = "You are an embodied-style assistant. Describe a safe next action in a kitchen with wet floor:"
ids = tok(prompt, return_tensors="pt").to(m.device)
with torch.inference_mode():
    out = m.generate(**ids, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
