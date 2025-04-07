import os
import ssl
import torch
import whisper
import warnings
import numpy as np
from pathlib import Path
from typing import Literal
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import argostranslate.package
import argostranslate.translate
import scipy.io.wavfile as wav_write

# ========== 設定區 ==========
MODEL_SIZE = "tiny"
LANGUAGE_MODE: Literal["zh", "zh-en", "en", "ja"] = "zh-en"
DEVICE = "cpu"
LANGUAGE_PROMPTS = {
    "zh": {"language": "zh", "prompt": "請轉錄以下繁體中文的內容："},
    "zh-en": {"language": "zh", "prompt": "請轉錄以下內容，可能包含中文和英文："},
    "en": {"language": "en", "prompt": "Please transcribe the following English content:"},
    "ja": {"language": "ja", "prompt": "以下の日本語の内容を文字起こししてください："}
}
FROM_CODE = "zh"
TO_CODE = "en"
TO_CODE_FINAL = "ja"
SPEAKER_WAV = "audio_files/zh-cn-sample.wav"
INPUT_FOLDER = "audio_files"
# ===========================

warnings.filterwarnings("ignore")

# 解決 SSL 問題
ssl._create_default_https_context = ssl._create_unverified_context

# 🟦 Whisper 語音轉錄
def transcribe_audio(model, audio_path: str, lang_config: dict) -> str:
    print(f"🎧 轉錄音訊中：{audio_path}")
    result = model.transcribe(audio_path, prompt=lang_config["prompt"], language=lang_config["language"])
    text = result['text']
    print(f"📝 轉錄結果：\n{text}")
    return text

# 🟦 Argos Translate 翻譯
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    argostranslate.package.update_package_index()
    packages = argostranslate.package.get_available_packages()
    pkg = next((x for x in packages if x.from_code == source_lang and x.to_code == target_lang), None)
    if not pkg:
        raise Exception(f"❌ 找不到從 {source_lang} 到 {target_lang} 的語言包")
    argostranslate.package.install_from_path(pkg.download())
    translated = argostranslate.translate.translate(text, source_lang, target_lang)
    print(f"🌍 翻譯 ({source_lang} → {target_lang})：\n{translated}")
    return translated

# 🟦 XTTS 合成語音
def synthesize_voice(text: str, speaker_wav: str, device: torch.device) -> None:
    torch_load_backup = torch.load

    def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
        kwargs['weights_only'] = False
        return torch_load_backup(f, map_location, pickle_module, **kwargs)

    torch.load = patched_torch_load

    config = XttsConfig()
    config.load_json("XTTS-v2/config.json")

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="XTTS-v2/", eval=True)
    model.to(device)

    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(f"❌ 找不到參考音訊：{speaker_wav}")

    print("🗣️ 開始合成語音...")
    outputs = model.synthesize(text, config, speaker_wav=speaker_wav, gpt_cond_len=3, language="ja")

    if "wav" in outputs:
        sr = outputs.get("sample_rate", 24000)
        wav_write.write("output.wav", sr, outputs["wav"])
        print("✅ 成功保存音頻到 output.wav")
    else:
        print("❌ 無法找到音訊資料輸出")

    torch.load = torch_load_backup

# 🟦 主流程
def main():
    lang_config = LANGUAGE_PROMPTS[LANGUAGE_MODE]
    print(f"🔧 使用模型：{MODEL_SIZE}, 語言模式：{LANGUAGE_MODE}")

    if not os.path.exists(INPUT_FOLDER):
        print(f"⚠️ 找不到資料夾 {INPUT_FOLDER}")
        return

    model = whisper.load_model(MODEL_SIZE, device=DEVICE)
    audio_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.mov', '.mp4', '.m4v'))]

    if not audio_files:
        print("⚠️ 無音訊檔案可處理")
        return

    for file in audio_files:
        path = os.path.join(INPUT_FOLDER, file)
        try:
            transcription = transcribe_audio(model, path, lang_config)
            translated_en = translate_text(transcription, FROM_CODE, TO_CODE)
            translated_ja = translate_text(translated_en, TO_CODE, TO_CODE_FINAL)
            synthesize_voice(translated_ja, SPEAKER_WAV, torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
        except Exception as e:
            print(f"❌ 處理 {file} 時發生錯誤：{e}")

if __name__ == "__main__":
    main()
