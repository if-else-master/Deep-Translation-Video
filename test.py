from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import functools
import types
import numpy as np
import os
import warnings

# 忽略警告（針對 GPT2InferenceModel 的警告）
warnings.filterwarnings("ignore")

# 用 monkey patching 替換 torch.load 函數
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, **pickle_load_args):
    # 總是使用 weights_only=False
    if 'weights_only' not in pickle_load_args:
        pickle_load_args['weights_only'] = False
    return original_torch_load(f, map_location, pickle_module, **pickle_load_args)

# 替換 torch.load 函數
torch.load = patched_torch_load

# 載入設定
config = XttsConfig()
config.load_json("XTTS-v2/config.json")

# 初始化模型
model = Xtts.init_from_config(config)

# 檢測並使用 MPS 裝置 (MacBook M4)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("使用 MPS 裝置 (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("MPS 不可用，使用 CPU")

try:
    # 載入檢查點
    model.load_checkpoint(config, checkpoint_dir="XTTS-v2/", eval=True)
    model.to(device)

    # 確保參考音頻檔案存在
    speaker_wav = "audio_files/ch.mp3"
    if not os.path.exists(speaker_wav):
        print(f"錯誤：參考音頻文件 {speaker_wav} 不存在")
        exit(1)

    # 合成語音
    print("開始合成語音...")
    outputs = model.synthesize(
        "大家好，這是一個測試。",  # 改用簡單的中文文本測試
        config,
        speaker_wav=speaker_wav,
        gpt_cond_len=3,
        language="zh-cn",  # 使用中文
    )
    
    print("合成完成，檢查輸出內容...")
    # 檢查輸出格式
    print(f"輸出包含的鍵: {list(outputs.keys())}")
    
    # 嘗試不同的鍵以保存音頻
    if "wav" in outputs and "sample_rate" in outputs:
        # 標準輸出格式
        import scipy.io.wavfile as wav_write
        wav_write.write("output.wav", outputs["sample_rate"], outputs["wav"])
        print("成功保存音頻到 output.wav")
    elif "wav" in outputs:
        # 只有波形數據，假設採樣率為 24000Hz (XTTS常用)
        import scipy.io.wavfile as wav_write
        wav_write.write("output.wav", 24000, outputs["wav"])
        print("成功保存音頻到 output.wav (使用預設採樣率 24000Hz)")
    else:
        # 直接保存第一個數組型數據
        for key, value in outputs.items():
            if isinstance(value, (np.ndarray, list)) and len(value) > 0:
                import scipy.io.wavfile as wav_write
                data = np.array(value)
                wav_write.write(f"output_{key}.wav", 24000, data)
                print(f"已保存找到的數據 '{key}' 到 output_{key}.wav")
                break
        else:
            print("無法在輸出中找到合適的音頻數據")
            print(f"輸出內容: {outputs}")
            
except Exception as e:
    print(f"發生錯誤: {e}")
    import traceback
    traceback.print_exc()

# 恢復原始 torch.load 函數
torch.load = original_torch_load