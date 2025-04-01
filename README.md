# Deep-Translation-Video

## 📌 專案簡介
**Deep-Translation-Video** 是一個基於多種開源技術的自動語音轉錄與翻譯系統。本專案結合 Whisper 進行語音轉錄、Argos Translate 進行多層翻譯，以及 Coqui XTTS-v2 進行語音合成，最終實現語音的跨語言翻譯與生成。

## 💻 本專案開發平台
- Python 3.10.11
- Numpy 1.22.0
- 電腦 Macbook Pro M4 Pro 24GB

## 🚀 主要功能
- 🎙 **語音轉錄**：使用 OpenAI Whisper 轉錄音訊，支援多種語言。
- 🌍 **語言翻譯**：利用 Argos Translate 進行多層語言翻譯。
- 🔊 **語音合成**：使用 Coqui XTTS-v2 生成翻譯後的語音。
- 📂 **影片嘴形調整**：使用 video-retalking 進行翻譯後影片嘴形的調整
## 📦 依賴項目
目前本專案基於以下開源技術：
- [Argos Translate](https://github.com/argosopentech/argos-translate)
- [Whisper](https://github.com/openai/whisper)
- [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2)

## 🛠 安裝與環境設定
### 1️⃣ 克隆專案與下載必要資源
```bash
python3 -m venv .venv
source .venv/bin/activate
```
```bash
git clone https://github.com/if-else-master/Deep-Translation-Video.git
cd Deep-Translation-Video
```

### 2️⃣ 安裝必要套件
```bash
pip install -r requirements.txt
```

### 3️⃣ 下載 Audio2Text-Whisper 模型
```bash
git clone https://github.com/sunny96087/Audio2Text.git
```
### 4️⃣ 下載 Argos-Translate 協助翻譯文本
```bash
git clone https://github.com/argosopentech/argos-translate.git
```

### 5️⃣ 下載 XTTS-v2
```bash
git clone https://huggingface.co/coqui/XTTS-v2
```

## 📂 使用方式
### 1️⃣ 將音訊檔案放入 `audio_files/` 資料夾
支援的格式：`.mp3`, `.wav`, `.m4a`, `.flac`, `.mov`, `.mp4`, `.m4v`

### 2️⃣ 執行轉錄與翻譯
```bash
python main.py
```

### 3️⃣ 生成翻譯後的語音
執行 `main.py` 後會產生翻譯文本，接著會自動合成音訊，輸出至 `output.wav`。

## ⚙️ 設定參數
本專案的 `main.py` 可根據需求調整：
```python
MODEL_SIZE = "tiny"  # Whisper 模型大小
LANGUAGE_MODE = "zh-en"  # 轉錄語言模式（可選 zh, zh-en, en, ja）
DEVICE = "cpu"  # 設定運行設備（"cpu" 或 "mps"）
```

## 📝 範例輸出
```plaintext
🎤 開始處理: example_audio.wav
🎧 轉錄音訊中...
📝 轉錄結果：大家好，這是一個測試。
🌍 翻譯 (第一層): 中文 → 英文
📝 英文翻譯結果：Hello, this is a test.
🌍 翻譯 (第二層): 英文 → 日文
📝 日文翻譯結果：こんにちは、これはテストです。
🔊 成功保存音頻到 output.wav
```

## ✅ TODO List
- [x] 將影片的內容輸出成逐字稿(`Open-AI：Audio2Text-Whisper`)
- [x] 將影片內容的逐字稿翻譯成不同語言的逐字稿(`Argos-Translate`)
- [x] 下載 XTTS-v2 協助克隆影片中的音頻及輸出 mp3 等檔案(`git clone https://huggingface.co/coqui/XTTS-v2`)
- [ ] 使用 Video-Retalking 進行影片翻譯後嘴型的校正 (`git clone https://github.com/OpenTalker/video-retalking.git`)
- [ ] 使用 Dango-Translator 翻譯影片中的其他文字 如：簡報、照片文字等 (`https://github.com/PantsuDango/Dango-Translator.git`)
- [ ] 專案 GUI 設計

## 📬 聯絡方式
Gmail：rayc57429@gmail.com





