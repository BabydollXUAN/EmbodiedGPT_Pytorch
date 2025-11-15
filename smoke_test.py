import os, sys, json
sys.path.append(os.path.dirname(__file__))

print("=== A. 包导入 ===")
import torch
import torchvision
import transformers
import decord
print("torch", torch.__version__, "| cuda?", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("decord", decord.__version__)

print("=== B. 数据管道 ===")
from robohusky.base_dataset_uni import BaseDataset
root = os.path.join(os.path.dirname(__file__), "datasets", "EgoCOT_base")
os.makedirs(root, exist_ok=True)
sample = os.path.join(root, "sample_list.json")
if not os.path.exists(sample):
    json.dump([{"image":"toy.jpg","caption":"toy"}], open(sample, "w"))
ds = BaseDataset(dataset=json.load(open(sample)), processor=None, image_path=root,
                 input_size=224, num_segments=4, norm_type="openai", media_type="image")
print("BaseDataset ok. len =", len(ds))

print("=== C. 极小模型前向（CPU）===")
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("distilgpt2")
m = AutoModelForCausalLM.from_pretrained("distilgpt2")
ids = tok("Say one safe action:", return_tensors="pt")
out = m.generate(**ids, max_new_tokens=12)
print(tok.decode(out[0], skip_special_tokens=True))
print("ALL GOOD ✓")
