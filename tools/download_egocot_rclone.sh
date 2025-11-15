# 用 rclone（你服务器那套）
#!/usr/bin/env bash
set -euo pipefail
mkdir -p ./datasets/EgoCOT_base
rclone copy gdrive: ./datasets/EgoCOT_base \
  --drive-root-folder-id "1d30x7S5MTz85JuqJcacQpp97T6Z2nbMt" \
  --drive-shared-with-me --transfers 8 --checkers 16 --progress --fast-list
