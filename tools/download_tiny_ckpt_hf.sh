# 用 HF 拉 tiny 权重
#!/usr/bin/env bash
set -euo pipefail
mkdir -p ./checkpoints/Embodied_family_7btiny
# 登录一次： huggingface-cli login
huggingface-cli download Liang-ZX/Embodied_family_7b \
  --local-dir ./checkpoints/Embodied_family_7btiny \
  --local-dir-use-symlinks False
