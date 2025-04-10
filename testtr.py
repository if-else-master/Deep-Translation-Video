import os
import subprocess


# 設定環境變數
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 構建指令
command = [
    "python3", "video-retalking/inference.py",
    "--face", "test/111.mp4",
    "--audio", "test/666.wav",
    "--outfile", "test/1_2.mp4"
]

# 執行指令
subprocess.run(command)
