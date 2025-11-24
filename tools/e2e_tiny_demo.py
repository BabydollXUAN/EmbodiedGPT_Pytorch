import os, sys, types, glob, inspect, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

# 给缺失的 tcsloader 一个占位，避免导入报错
shim = types.ModuleType("robohusky.train.tcsloader")
class TCSLoader: pass
shim.TCSLoader = TCSLoader
sys.modules["robohusky.train.tcsloader"] = shim

from transformers import AutoTokenizer, AutoModelForCausalLM

# 尝试优先用项目里的 BaseDataset；若失败就走兜底
def try_make_basedataset(root):
    try:
        from robohusky.base_dataset_uni import BaseDataset
    except Exception as e:
        print("[WARN] cannot import BaseDataset, fallback to raw npy:", e)
        return None

    sig = inspect.signature(BaseDataset)
    params = set(sig.parameters)
    kw = {}
    # 常见参数名里找一个可用的
    for key in ["root_dir", "root", "data_root", "data_dir", "dataset_root", "base_dir", "base_path"]:
        if key in params:
            kw[key] = root
            break
    # split_json 如果被支持就带上
    sj = os.path.join(root, "sample_list.json")
    if "split_json" in params and os.path.exists(sj):
        kw["split_json"] = sj
    try:
        return BaseDataset(**kw)
    except Exception as e:
        print("[WARN] BaseDataset(**%s) failed: %s" % (kw, e))
        return None

def load_samples(root, n=4):
    ds = try_make_basedataset(root)
    if ds is not None and len(ds) > 0:
        idx = list(range(min(n, len(ds))))
        return [ds[i] for i in idx]

    # 兜底：直接读取若干 .npy 文件，做成最简 dict
    files = sorted(glob.glob(os.path.join(root, "EGO_*.npy")))
    if not files:
        raise RuntimeError(f"No samples found under {root}")
    items = []
    for f in files[:n]:
        try:
            # 大文件别真的 load，这里只示例；真实项目你按需要加载
            items.append({"_path": f, "instruction": f"Describe content of {os.path.basename(f)}"})
        except Exception:
            items.append({"_path": f, "instruction": "Describe the scene."})
    return items

def to_prompt(item):
    for k in ["instruction","question","inst","text","caption","goal"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return f"Instruction: {v.strip()}\nAnswer:"
    return "Describe the scene and suggest the next robot action.\nAnswer:"

def main():
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    CKPT = "./checkpoints/Embodied_family_7btiny"
    DATA = "./datasets/EgoCOT_base"
    N    = 4

    tok  = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)
    mdl  = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.float16, local_files_only=True).cuda().eval()

    items = load_samples(DATA, N)
    print(f"[OK] loaded {len(items)} samples")

    for i, it in enumerate(items):
        prompt = to_prompt(it)
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = mdl.generate(ids, max_new_tokens=32)
        gen = tok.decode(out[0], skip_special_tokens=True)
        print(f"\n=== SAMPLE {i} ===")
        print("PROMPT:", prompt)
        print("GEN   :", gen)

if __name__ == "__main__":
    main()
