# tools/e2e_tiny_demo.py
import os, sys, types, glob, inspect, json, argparse
import torch

# 允许离线 & 固定缓存目录（可选）
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.path.abspath("./.hf_home"))

# 让 `import robohusky.train.tcsloader` 不报错
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
shim = types.ModuleType("robohusky.train.tcsloader")
class TCSLoader: pass
shim.TCSLoader = TCSLoader
sys.modules["robohusky.train.tcsloader"] = shim

from transformers import AutoTokenizer, AutoModelForCausalLM

def try_make_basedataset(root):
    """
    优先尝试项目的数据集类；不行就走兜底（直接读 .npy 列表）
    """
    try:
        from robohusky.base_dataset_uni import BaseDataset
    except Exception as e:
        print("[WARN] cannot import BaseDataset, fallback to raw npy:", e)
        return None

    sig = inspect.signature(BaseDataset)
    params = set(sig.parameters)
    kw = {}

    # 常见 root 参数名里挑一个
    for key in ["root_dir", "root", "data_root", "data_dir", "dataset_root", "base_dir", "base_path"]:
        if key in params:
            kw[key] = root
            break

    # 可选的 split_json
    sj = os.path.join(root, "sample_list.json")
    if "split_json" in params and os.path.exists(sj):
        kw["split_json"] = sj

    # 某些实现需要 dataset / processor，占位给个默认
    if "dataset" in params and "dataset" not in kw:
        kw["dataset"] = "EgoCOT"
    if "processor" in params and "processor" not in kw:
        kw["processor"] = None

    try:
        ds = BaseDataset(**kw)
        return ds
    except Exception as e:
        print(f"[WARN] BaseDataset(**{kw}) failed: {e}")
        return None

def load_samples(root, n=4):
    ds = try_make_basedataset(root)
    if ds is not None:
        try:
            n = min(n, len(ds))
            if n > 0:
                return [ds[i] for i in range(n)]
        except Exception as e:
            print("[WARN] dataset read failed, fallback to npy:", e)

    # 兜底：直接读取若干 .npy 文件，做成最简 dict
    files = sorted(glob.glob(os.path.join(root, "EGO_*.npy")))
    if not files:
        raise RuntimeError(f"No samples found under {root}")
    items = []
    for f in files[:n]:
        items.append({
            "_path": f,
            "instruction": f"Describe content of {os.path.basename(f)}"
        })
    return items

def to_prompt(item):
    # 从常见字段里挑一个 prompt 字段；否则给默认
    for k in ["instruction", "question", "inst", "text", "caption", "goal"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return f"Instruction: {v.strip()}\nAnswer:"
    # 兜底
    name = os.path.basename(item.get("_path", "the scene"))
    return f"Instruction: Describe content of {name}\nAnswer:"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="./checkpoints/Embodied_family_7btiny", help="tiny 权重目录（本地离线）")
    parser.add_argument("--data", default="./datasets/EgoCOT_base", help="EgoCOT base 数据目录")
    parser.add_argument("-n", "--num", type=int, default=4, help="示例条数")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # 加载分词器与模型（本地离线）
    tok = AutoTokenizer.from_pretrained(args.ckpt, local_files_only=True)
    # 解决 pad_token 警告：用 eos 作为 pad
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=dtype, local_files_only=True
    ).to(device).eval()

    items = load_samples(args.data, args.num)
    print(f"[OK] loaded {len(items)} samples from: {args.data}")

    for i, it in enumerate(items):
        prompt = to_prompt(it)
        enc = tok(prompt, return_tensors="pt", padding=True)
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device) if "attention_mask" in enc else None

        with torch.no_grad():
            out = mdl.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,          # 更稳定复现；想要发散就设 True + 温度
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0], skip_special_tokens=True)

        print(f"\n=== SAMPLE {i} ===")
        print("PROMPT:", prompt)
        print("GEN   :", gen)

if __name__ == "__main__":
    main()

