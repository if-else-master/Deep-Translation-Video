import torch
import whisper
import os
import ssl
from pathlib import Path
from tqdm import tqdm
from typing import Literal
import argostranslate.package
import argostranslate.translate

# 使用者設定區域 👩🏻‍💻 ======================
MODEL_SIZE = "tiny"
LANGUAGE_MODE = "zh-en"

DEVICE = "cpu"  # 強制使用 CPU，避免 MPS 相容性問題

LANGUAGE_PROMPTS = {
    "zh": {"language": "zh", "prompt": "請轉錄以下繁體中文的內容："},
    "zh-en": {"language": "zh", "prompt": "請轉錄以下內容，可能包含中文和英文："},
    "en": {"language": "en", "prompt": "Please transcribe the following English content:"},
    "ja": {"language": "ja", "prompt": "以下の日本語の内容を文字起こししてください："}
}

# 翻譯語言
from_code = "zh"  # 轉錄結果的語言
to_code = "en"    # 第一層翻譯
to_code_final = "ja"  # 第二層翻譯

# ===========================================

# 全局變量
transcription_result = ""
transcription_en = ""

def transcribe_audio_files():
    global transcription_result

    ssl._create_default_https_context = ssl._create_unverified_context
    lang_config = LANGUAGE_PROMPTS.get(LANGUAGE_MODE, LANGUAGE_PROMPTS["zh"])

    print(f"使用模型: {MODEL_SIZE}")
    print(f"語言模式: {LANGUAGE_MODE}")
    print(f"語言提示: {lang_config['prompt']}")

    cache_dir = Path("./whisper_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)

    input_folder = "audio_files"
    if not os.path.exists(input_folder):
        print(f"⚠️ 資料夾 '{input_folder}' 不存在，請確認音訊檔案路徑")
        return

    print("載入 Whisper 模型中...")
    model = whisper.load_model(MODEL_SIZE, device=DEVICE)

    audio_extensions = ('.mp3', '.wav', '.m4a', '.flac', '.mov', '.mp4', '.m4v')
    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(audio_extensions)]

    if not audio_files:
        print(f"⚠️ '{input_folder}' 資料夾中沒有找到音訊檔案")
        return

    for audio_file in audio_files:
        print(f"\n🎤 開始處理: {audio_file}")
        input_path = os.path.join(input_folder, audio_file)

        try:
            print("🎧 轉錄音訊中...")
            result = model.transcribe(input_path, prompt=lang_config["prompt"], language=lang_config["language"])
            transcription_result = result['text']
            print(f"📝 轉錄結果：\n{transcription_result}")

        except Exception as e:
            print(f"❌ 處理 {audio_file} 時發生錯誤: {str(e)}")

def translate_transcription():
    global transcription_result, transcription_en

    print("\n🌍 翻譯 (第一層): 中文 → 英文")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    package_to_install = next(
        (x for x in available_packages if x.from_code == from_code and x.to_code == to_code),
        None
    )

    if package_to_install is None:
        print(f"❌ 未找到從 {from_code} 到 {to_code} 的語言包")
        return

    argostranslate.package.install_from_path(package_to_install.download())

    # 進行翻譯
    transcription_en = argostranslate.translate.translate(transcription_result, from_code, to_code)
    print(f"📝 英文翻譯結果：\n{transcription_en}")

def translate_transcription_to():
    global transcription_en

    print("\n🌍 翻譯 (第二層): 英文 → 日文")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()

    package_to_install = next(
        (x for x in available_packages if x.from_code == to_code and x.to_code == to_code_final),
        None
    )

    if package_to_install is None:
        print(f"❌ 未找到從 {to_code} 到 {to_code_final} 的語言包")
        return

    argostranslate.package.install_from_path(package_to_install.download())

    # 進行翻譯
    translated_text = argostranslate.translate.translate(transcription_en, to_code, to_code_final)
    print(f"📝 日文翻譯結果：\n{translated_text}")

if __name__ == "__main__":
    transcribe_audio_files()
    translate_transcription()
    translate_transcription_to()
