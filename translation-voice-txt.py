import os
import ssl
import torch
import whisper
import warnings
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
from pathlib import Path
from typing import Literal
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import argostranslate.package
import argostranslate.translate
import scipy.io.wavfile as wav_write
import threading
import pygame

# 禁用警告並設置SSL上下文
warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context

# 語言設置
LANGUAGE_PROMPTS = {
    "zh": {"language": "zh", "prompt": "請轉錄以下繁體中文的內容：", "display": "中文"},
    "zh-en": {"language": "zh", "prompt": "請轉錄以下內容，可能包含中文和英文：", "display": "中文和英文"},
    "en": {"language": "en", "prompt": "Please transcribe the following English content:", "display": "英文"},
    "ja": {"language": "ja", "prompt": "以下の日本語の内容を文字起こししてください：", "display": "日文"}
}

# 支持的語言代碼
LANGUAGE_CODES = {
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "韓文": "ko",
    "法文": "fr",
    "德文": "de",
    "西班牙文": "es",
    "俄文": "ru"
}

# 模型大小選項
MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]

class AudioProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多語言音訊處理器")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")

        # 強制使用CPU而非MPS
        # 禁用MPS和CUDA，避免後端問題
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        # 初始化pygame來播放音訊
        pygame.mixer.init()
        
        # 創建主框架
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 音訊檔案選擇區域
        file_frame = ttk.LabelFrame(main_frame, text="音訊檔案", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        self.audio_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.audio_path_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="瀏覽", command=self.browse_audio_file).pack(side=tk.LEFT, padx=5)
        
        # 參考語音選擇區域
        speaker_frame = ttk.LabelFrame(main_frame, text="參考語音", padding=10)
        speaker_frame.pack(fill=tk.X, pady=5)
        
        self.speaker_path_var = tk.StringVar()
        ttk.Entry(speaker_frame, textvariable=self.speaker_path_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(speaker_frame, text="瀏覽", command=self.browse_speaker_file).pack(side=tk.LEFT, padx=5)
        
        # 語言和模型配置區域
        config_frame = ttk.LabelFrame(main_frame, text="配置", padding=10)
        config_frame.pack(fill=tk.X, pady=5)
        
        # 左側面板
        left_config = ttk.Frame(config_frame)
        left_config.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 模型大小
        ttk.Label(left_config, text="Whisper 模型大小:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_size_var = tk.StringVar(value="tiny")
        model_combo = ttk.Combobox(left_config, textvariable=self.model_size_var, values=MODEL_SIZES, state="readonly", width=15)
        model_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 轉錄語言模式
        ttk.Label(left_config, text="轉錄語言模式:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.lang_mode_var = tk.StringVar(value="zh-en")
        lang_combo = ttk.Combobox(left_config, textvariable=self.lang_mode_var, 
                                  values=list(LANGUAGE_PROMPTS.keys()), 
                                  state="readonly", width=15)
        lang_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # 右側面板
        right_config = ttk.Frame(config_frame)
        right_config.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 源語言
        ttk.Label(right_config, text="源語言:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.from_lang_var = tk.StringVar(value="中文")
        from_combo = ttk.Combobox(right_config, textvariable=self.from_lang_var, 
                                   values=list(LANGUAGE_CODES.keys()), 
                                   state="readonly", width=15)
        from_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 目標語言1
        ttk.Label(right_config, text="中間翻譯語言:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.to_lang_var = tk.StringVar(value="英文")
        to_combo = ttk.Combobox(right_config, textvariable=self.to_lang_var, 
                                 values=list(LANGUAGE_CODES.keys()), 
                                 state="readonly", width=15)
        to_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # 目標語言2
        ttk.Label(right_config, text="最終翻譯語言:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.final_lang_var = tk.StringVar(value="日文")
        final_combo = ttk.Combobox(right_config, textvariable=self.final_lang_var, 
                                    values=list(LANGUAGE_CODES.keys()), 
                                    state="readonly", width=15)
        final_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 裝置選擇
        device_frame = ttk.Frame(config_frame)
        device_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        ttk.Label(device_frame, text="處理裝置:").pack(side=tk.LEFT, padx=5)
        self.device_var = tk.StringVar(value="cpu")
        
        # 只提供CPU選項，強制使用CPU以避免MPS問題
        device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                     values=["cpu"], state="readonly", width=10)
        device_combo.pack(side=tk.LEFT, padx=5)
        
        # 輸出區域
        output_frame = ttk.LabelFrame(main_frame, text="處理結果", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 使用Notebook創建標籤頁
        self.output_tabs = ttk.Notebook(output_frame)
        self.output_tabs.pack(fill=tk.BOTH, expand=True)
        
        # 轉錄標籤頁
        trans_tab = ttk.Frame(self.output_tabs)
        self.output_tabs.add(trans_tab, text="轉錄")
        self.transcription_text = scrolledtext.ScrolledText(trans_tab, wrap=tk.WORD, height=10)
        self.transcription_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 翻譯1標籤頁
        trans1_tab = ttk.Frame(self.output_tabs)
        self.output_tabs.add(trans1_tab, text="中間翻譯")
        self.translation1_text = scrolledtext.ScrolledText(trans1_tab, wrap=tk.WORD, height=10)
        self.translation1_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 翻譯2標籤頁
        trans2_tab = ttk.Frame(self.output_tabs)
        self.output_tabs.add(trans2_tab, text="最終翻譯")
        self.translation2_text = scrolledtext.ScrolledText(trans2_tab, wrap=tk.WORD, height=10)
        self.translation2_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 日誌標籤頁
        log_tab = ttk.Frame(self.output_tabs)
        self.output_tabs.add(log_tab, text="處理日誌")
        self.log_text = scrolledtext.ScrolledText(log_tab, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 按鈕區域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.progress = ttk.Progressbar(button_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="就緒")
        status_label = ttk.Label(button_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=5)
        
        self.process_btn = ttk.Button(button_frame, text="開始處理", command=self.start_processing)
        self.process_btn.pack(side=tk.RIGHT, padx=5)
        
        self.play_btn = ttk.Button(button_frame, text="播放輸出", command=self.play_output, state=tk.DISABLED)
        self.play_btn.pack(side=tk.RIGHT, padx=5)
        
        # 模型加載狀態
        self.whisper_model = None
        
        # 初始化日誌
        self.log("應用程序已啟動，準備就緒")
        self.log("注意：已強制使用CPU模式以確保兼容性")
    
    def log(self, message):
        """添加日誌訊息"""
        self.log_text.insert(tk.END, f"[INFO] {message}\n")
        self.log_text.see(tk.END)
        print(message)
    
    def browse_audio_file(self):
        """瀏覽並選擇音訊檔案"""
        file_path = filedialog.askopenfilename(
            title="選擇音訊檔案",
            filetypes=[("音訊檔案", "*.wav *.mp3 *.ogg"), ("所有檔案", "*.*")]
        )
        if file_path:
            self.audio_path_var.set(file_path)
            self.log(f"已選擇音訊檔案: {file_path}")
            # 自動設置同一個檔案為參考語音
            self.speaker_path_var.set(file_path)
    
    def browse_speaker_file(self):
        """瀏覽並選擇參考語音檔案"""
        file_path = filedialog.askopenfilename(
            title="選擇參考語音檔案",
            filetypes=[("音訊檔案", "*.wav *.mp3 *.ogg"), ("所有檔案", "*.*")]
        )
        if file_path:
            self.speaker_path_var.set(file_path)
            self.log(f"已選擇參考語音: {file_path}")
    
    def update_status(self, message):
        """更新狀態訊息"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def start_processing(self):
        """啟動處理線程"""
        audio_path = self.audio_path_var.get()
        speaker_path = self.speaker_path_var.get()
        
        if not audio_path:
            self.log("❌ 請選擇音訊檔案")
            return
        
        if not speaker_path:
            self.log("❌ 請選擇參考語音檔案")
            return
        
        # 禁用按鈕並顯示進度條
        self.process_btn.configure(state=tk.DISABLED)
        self.play_btn.configure(state=tk.DISABLED)
        self.progress.start()
        self.update_status("處理中...")
        
        # 創建新線程來處理音訊
        process_thread = threading.Thread(target=self.process_audio)
        process_thread.daemon = True
        process_thread.start()
    
    def process_audio(self):
        """處理音訊的主要流程"""
        try:
            # 獲取配置
            audio_path = self.audio_path_var.get()
            speaker_path = self.speaker_path_var.get()
            model_size = self.model_size_var.get()
            lang_mode = self.lang_mode_var.get()
            from_lang_code = LANGUAGE_CODES[self.from_lang_var.get()]
            to_lang_code = LANGUAGE_CODES[self.to_lang_var.get()]
            final_lang_code = LANGUAGE_CODES[self.final_lang_var.get()]
            
            self.log(f"🔧 使用模型: {model_size}, 語言模式: {lang_mode}")
            self.log(f"🔧 翻譯路徑: {from_lang_code} → {to_lang_code} → {final_lang_code}")
            
            # 始終使用CPU設備
            device = torch.device("cpu")
            self.log(f"🔄 使用設備: {device} (已強制使用CPU以避免MPS問題)")
            
            # 載入Whisper模型
            self.log(f"🔄 正在載入Whisper模型 ({model_size})...")
            model = whisper.load_model(model_size, device=device)
            
            # 轉錄音訊
            self.log(f"🎧 轉錄音訊中: {audio_path}")
            lang_config = LANGUAGE_PROMPTS[lang_mode]
            result = model.transcribe(audio_path, prompt=lang_config["prompt"], language=lang_config["language"])
            transcription = result['text']
            
            # 更新UI
            self.root.after(0, lambda: self.transcription_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.transcription_text.insert(tk.END, transcription))
            self.log(f"📝 已完成轉錄")
            
            # 翻譯文本 (第一次)
            self.log(f"🌍 翻譯中 ({from_lang_code} → {to_lang_code})...")
            translated_middle = self.translate_text(transcription, from_lang_code, to_lang_code)
            
            # 更新UI
            self.root.after(0, lambda: self.translation1_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.translation1_text.insert(tk.END, translated_middle))
            
            # 翻譯文本 (第二次)
            self.log(f"🌍 翻譯中 ({to_lang_code} → {final_lang_code})...")
            translated_final = self.translate_text(translated_middle, to_lang_code, final_lang_code)
            
            # 更新UI
            self.root.after(0, lambda: self.translation2_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.translation2_text.insert(tk.END, translated_final))
            
            # 合成語音
            self.log("🗣️ 開始合成語音...")
            self.synthesize_voice(translated_final, speaker_path, device)
            
            # 完成處理
            self.log("✅ 全部處理完成")
            self.root.after(0, lambda: self.update_status("處理完成"))
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.process_btn.configure(state=tk.NORMAL))
            self.root.after(0, lambda: self.play_btn.configure(state=tk.NORMAL))
            
        except Exception as e:
            error_msg = str(e)
            self.log(f"❌ 處理過程中發生錯誤: {error_msg}")
            
            # 為常見錯誤提供更友好的提示
            if "CUDA" in error_msg or "MPS" in error_msg:
                self.log("💡 提示: 這可能是GPU加速相關問題，程序已強制使用CPU模式")
            elif "load_checkpoint" in error_msg:
                self.log("💡 提示: 請確認XTTS-v2模型檔案路徑正確，並包含所有必要檔案")
            elif "wav" in error_msg.lower() or "audio" in error_msg.lower():
                self.log("💡 提示: 音訊檔案可能格式不兼容，請嘗試使用標準WAV格式")
            
            self.root.after(0, lambda: self.update_status("處理失敗"))
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.process_btn.configure(state=tk.NORMAL))
    
    def translate_text(self, text, source_lang, target_lang):
        """使用Argos翻譯文本"""
        try:
            self.log(f"🔄 檢查和安裝語言包 {source_lang} → {target_lang}...")
            argostranslate.package.update_package_index()
            packages = argostranslate.package.get_available_packages()
            package_found = False
            
            for pkg in packages:
                if hasattr(pkg, "from_code") and hasattr(pkg, "to_code") and pkg.from_code == source_lang and pkg.to_code == target_lang:
                    self.log(f"🔄 正在安裝語言包: {source_lang} → {target_lang}")
                    argostranslate.package.install_from_path(pkg.download())
                    package_found = True
                    break
            
            if not package_found:
                raise Exception(f"❌ 找不到從 {source_lang} 到 {target_lang} 的語言包")
            
            self.log(f"🔄 正在翻譯文本...")
            translated = argostranslate.translate.translate(text, source_lang, target_lang)
            return translated
        except Exception as e:
            self.log(f"❌ 翻譯過程中發生錯誤: {str(e)}")
            raise e
    
    def synthesize_voice(self, text, speaker_wav, device):
        """使用XTTS合成語音"""
        try:
            # 確保XTTS目錄存在
            xtts_dir = "XTTS-v2"
            if not os.path.exists(xtts_dir):
                self.log(f"❌ 找不到XTTS模型目錄: {xtts_dir}")
                self.log("💡 提示: 請確保已下載XTTS-v2模型並放置在正確位置")
                raise FileNotFoundError(f"找不到XTTS模型目錄: {xtts_dir}")
            
            config_path = os.path.join(xtts_dir, "config.json")
            if not os.path.exists(config_path):
                self.log(f"❌ 找不到XTTS配置文件: {config_path}")
                raise FileNotFoundError(f"找不到XTTS配置文件: {config_path}")
            
            # 備份原始torch.load函數
            torch_load_backup = torch.load
            
            # 修補torch.load函數
            def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
                kwargs['weights_only'] = False
                return torch_load_backup(f, map_location, pickle_module, **kwargs)
            
            # 設置load函數
            torch.load = patched_torch_load
            
            # 配置XTTS
            try:
                self.log("🔄 載入XTTS配置...")
                config = XttsConfig()
                config.load_json(config_path)
            except Exception as e:
                self.log(f"❌ 載入XTTS配置時出錯: {str(e)}")
                raise e
            
            # 載入XTTS模型
            try:
                self.log("🔄 正在載入XTTS模型...")
                model = Xtts.init_from_config(config)
                model.load_checkpoint(config, checkpoint_dir=xtts_dir, eval=True)
                model.to(device)
            except Exception as e:
                self.log(f"❌ 載入XTTS模型時出錯: {str(e)}")
                self.log("💡 提示: 請確保模型檔案完整且未損壞")
                raise e
            
            if not os.path.exists(speaker_wav):
                self.log(f"❌ 找不到參考音訊: {speaker_wav}")
                raise FileNotFoundError(f"找不到參考音訊: {speaker_wav}")
            
            # 最終語言代碼
            final_lang_code = LANGUAGE_CODES[self.final_lang_var.get()]
            
            # 生成合成語音
            try:
                self.log("🔊 正在生成合成語音...")
                self.log(f"🔊 使用語言: {final_lang_code}, 參考音訊: {os.path.basename(speaker_wav)}")
                
                outputs = model.synthesize(
                    text=text,
                    config=config,
                    speaker_wav=speaker_wav,
                    gpt_cond_len=3,
                    language=final_lang_code
                )
                
                if "wav" in outputs:
                    sr = outputs.get("sample_rate", 24000)
                    output_path = "output.wav"
                    wav_write.write(output_path, sr, outputs["wav"])
                    self.log(f"✅ 成功保存音頻到 {output_path}")
                else:
                    self.log("❌ 無法找到音訊資料輸出")
                    raise Exception("合成過程未生成有效的音訊資料")
            except Exception as e:
                self.log(f"❌ 音訊合成時出錯: {str(e)}")
                raise e
            finally:
                # 恢復原始torch.load函數
                torch.load = torch_load_backup
            
        except Exception as e:
            self.log(f"❌ 語音合成過程中發生錯誤: {str(e)}")
            raise e
    
    def play_output(self):
        """播放生成的音訊輸出"""
        output_path = "output.wav"
        if os.path.exists(output_path):
            try:
                pygame.mixer.music.load(output_path)
                pygame.mixer.music.play()
                self.log("🎵 正在播放合成的音訊...")
            except Exception as e:
                self.log(f"❌ 播放音訊時發生錯誤: {str(e)}")
        else:
            self.log(f"❌ 找不到輸出音訊檔案: {output_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioProcessorApp(root)
    root.mainloop()