import os
import ssl
import torch
import whisper
import warnings
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from pathlib import Path
from typing import Literal
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import argostranslate.package
import argostranslate.translate
import scipy.io.wavfile as wav_write
import threading
import pygame
from pydub import AudioSegment
import subprocess
import shutil
import tempfile
import moviepy as mp
from datetime import datetime


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

# 輸出格式選項
AUDIO_FORMATS = {
    "WAV": {"ext": "wav", "display": "WAV (無損)"},
    "MP3": {"ext": "mp3", "display": "MP3 (常用格式)"},
    "M4A": {"ext": "m4a", "display": "M4A (Apple格式)"},
    "OGG": {"ext": "ogg", "display": "OGG (開放格式)"}
}

# 視頻格式選項
VIDEO_FORMATS = {
    "MP4": {"ext": "mp4", "display": "MP4 (常用格式)"},
    "MOV": {"ext": "mov", "display": "MOV (Apple格式)"},
    "MKV": {"ext": "mkv", "display": "MKV (開放格式)"}
}

# 媒體類型
MEDIA_TYPES = {
    "AUDIO": "音訊",
    "VIDEO": "視頻"
}

class AudioProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多語言媒體處理器")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")

        # 強制使用CPU而非MPS
        # 禁用MPS和CUDA，避免後端問題
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        # 初始化pygame來播放音訊
        pygame.mixer.init()
        
        # 臨時文件和狀態追踪
        self.temp_files = []  # 保存程序過程中創建的臨時文件
        self.input_media_type = None  # 輸入媒體類型 (音訊/視頻)
        self.extracted_audio_path = None  # 從視頻中提取的音訊路徑
        
        # 創建主框架
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 輸入檔案選擇區域
        file_frame = ttk.LabelFrame(main_frame, text="輸入檔案", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        
        self.audio_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.audio_path_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(file_frame, text="瀏覽", command=self.browse_input_file).pack(side=tk.LEFT, padx=5)
        browse_folder_button = ttk.Button(file_frame, text="選擇資料夾批量處理", command=self.browse_input_folder)
        browse_folder_button.pack(side=tk.LEFT, padx=5)
        #folder_path = filedialog.askdirectory(title="選擇資料夾")
        
        # 輸入文件類型顯示
        self.input_type_var = tk.StringVar(value="未選擇檔案")
        input_type_label = ttk.Label(file_frame, textvariable=self.input_type_var)
        input_type_label.pack(side=tk.LEFT, padx=5)
        
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
        
        # 輸出格式選擇
        ttk.Label(device_frame, text="輸出格式:").pack(side=tk.LEFT, padx=5)
        self.format_var = tk.StringVar(value="WAV")
        format_values = [f"{k} - {v['display']}" for k, v in AUDIO_FORMATS.items()]
        format_combo = ttk.Combobox(device_frame, textvariable=self.format_var, 
                                    values=format_values, state="readonly", width=20)
        format_combo.pack(side=tk.LEFT, padx=5)
        
        # 輸出媒體類型選擇
        ttk.Label(device_frame, text="輸出類型:").pack(side=tk.LEFT, padx=5)
        self.output_type_var = tk.StringVar(value="AUDIO")
        output_type_combo = ttk.Combobox(device_frame, textvariable=self.output_type_var, 
                                        values=["AUDIO - 僅輸出音訊", "VIDEO - 輸出視頻"], 
                                        state="readonly", width=20)
        output_type_combo.pack(side=tk.LEFT, padx=5)
        
        # 當輸出類型變更時更新格式選項
        def update_format_options(*args):
            output_type = self.output_type_var.get().split(" - ")[0]
            if output_type == "AUDIO":
                format_combo['values'] = [f"{k} - {v['display']}" for k, v in AUDIO_FORMATS.items()]
                if not any(self.format_var.get().startswith(k) for k in AUDIO_FORMATS.keys()):
                    self.format_var.set("WAV - WAV (無損)")
            else:  # VIDEO
                format_combo['values'] = [f"{k} - {v['display']}" for k, v in VIDEO_FORMATS.items()]
                if not any(self.format_var.get().startswith(k) for k in VIDEO_FORMATS.keys()):
                    self.format_var.set("MP4 - MP4 (常用格式)")
        
        # 設置變更監聽
        self.output_type_var.trace_add("write", update_format_options)
        
        # 初始化
        update_format_options()
        
        # 輸出區域
        output_frame = ttk.LabelFrame(main_frame, text="處理結果", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 使用Notebook創建標籤頁
        self.output_tabs = ttk.Notebook(output_frame)
        self.output_tabs.pack(fill=tk.BOTH, expand=True)
        # 日誌標籤頁
        log_tab = ttk.Frame(self.output_tabs)
        self.output_tabs.add(log_tab, text="處理日誌")
        self.log_text = scrolledtext.ScrolledText(log_tab, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
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
        
        self.save_btn = ttk.Button(button_frame, text="另存為", command=self.save_as, state=tk.DISABLED)
        self.save_btn.pack(side=tk.RIGHT, padx=5)
        
        # 添加視頻換臉按鈕
        self.retalk_btn = ttk.Button(button_frame, text="視頻換臉", command=self.start_video_retalk, state=tk.DISABLED)
        self.retalk_btn.pack(side=tk.RIGHT, padx=5)
        
        # 模型加載狀態
        self.whisper_model = None
        
        # 當前輸出音訊路徑
        self.current_output_path = "output.wav"
        
        # 初始化日誌
        self.log("應用程序已啟動，準備就緒")
        self.log("注意：已強制使用CPU模式以確保兼容性")
        
        # 檢查 FFmpeg 是否可用（用於音訊格式轉換）
        self.check_ffmpeg()
    
    def log(self, message):
        """添加日誌訊息"""
        self.log_text.insert(tk.END, f"[INFO] {message}\n")
        self.log_text.see(tk.END)
        print(message)
    def browse_input_folder(self):
        folder_path = filedialog.askdirectory(title="選擇資料夾")
        if not folder_path:
            return

        # 支援的副檔名
        supported_exts = ('.wav', '.mp3', '.ogg', '.m4a', '.mp4', '.mov', '.mkv')

        # 取得符合的檔案列表
        files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(supported_exts)
        ]

        # 逐一處理檔案
        for file_path in files:
            if file_path:
                self.audio_path_var.set(file_path)
                self.log(f"已選擇音訊檔案: {file_path}")
                # 確定文件類型
                file_ext = os.path.splitext(file_path)[1].lower()                
                # 清理之前可能存在的臨時檔案    
                self.cleanup_temp_files()                
                # 自動設置同一個檔案為參考語音
                self.speaker_path_var.set(file_path)                

    def browse_input_file(self):
        """瀏覽並選擇輸入檔案（音訊或視頻）"""
        file_path = filedialog.askopenfilename(
            title="選擇輸入檔案",
            filetypes=[
                ("所有支援的媒體檔案", "*.wav *.mp3 *.ogg *.m4a *.mp4 *.mov *.mkv"),
                ("音訊檔案", "*.wav *.mp3 *.ogg *.m4a"),
                ("視頻檔案", "*.mp4 *.mov *.mkv"),
                ("所有檔案", "*.*")
            ]
        )        
        if file_path:
            self.audio_path_var.set(file_path)
            
            # 確定文件類型
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 清理之前可能存在的臨時檔案
            self.cleanup_temp_files()
            
            if file_ext in ['.mp4', '.mov', '.mkv']:
                # 這是視頻文件
                self.input_media_type = MEDIA_TYPES["VIDEO"]
                self.input_type_var.set(f"視頻檔案 ({file_ext[1:].upper()})")
                self.log(f"已選擇視頻檔案: {file_path}")
                
                # 從視頻中提取音訊
                self.extract_audio_from_video(file_path)
            else:
                # 這是音訊文件
                self.input_media_type = MEDIA_TYPES["AUDIO"]
                self.input_type_var.set(f"音訊檔案 ({file_ext[1:].upper()})")
                self.log(f"已選擇音訊檔案: {file_path}")
                
                # 自動設置同一個檔案為參考語音
                self.speaker_path_var.set(file_path)
    
    def extract_audio_from_video(self, video_path):
        """從視頻檔案中提取音訊"""
        try:
            self.log(f"🔄 正在從視頻中提取音訊...")
            
            # 創建臨時目錄管理臨時檔案
            temp_dir = tempfile.mkdtemp()
            temp_audio_path = os.path.join(temp_dir, "extracted_audio.wav")
            
            # 使用 moviepy 提取音訊
            video = mp.VideoFileClip(video_path)
            video.audio.write_audiofile(temp_audio_path, logger=None)
            
            # 保存路徑供後續處理
            self.extracted_audio_path = temp_audio_path
            self.temp_files.append(temp_dir)  # 添加到臨時文件列表以便之後清理
            
            # 自動設置提取的音訊為參考語音
            self.speaker_path_var.set(temp_audio_path)
            
            self.log(f"✅ 成功從視頻中提取音訊")
            
        except Exception as e:
            self.log(f"❌ 從視頻提取音訊時發生錯誤: {str(e)}")
    
    def cleanup_temp_files(self):
        """清理臨時文件和目錄"""
        for temp_path in self.temp_files:
            try:
                if os.path.isdir(temp_path):
                    shutil.rmtree(temp_path)
                elif os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                self.log(f"⚠️ 清理臨時文件時出錯: {str(e)}")
        
        # 重置臨時文件列表
        self.temp_files = []
        self.extracted_audio_path = None
    
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
            self.log("❌ 請選擇輸入檔案")
            return
        
        if not speaker_path:
            self.log("❌ 請選擇參考語音檔案")
            return
        
        # 檢查所選輸出類型和輸入文件是否兼容
        output_type = self.output_type_var.get().split(" - ")[0]
        
        if output_type == "VIDEO" and self.input_media_type == MEDIA_TYPES["VIDEO"]:
            # 檢查源視頻是否有音訊
            try:
                video = mp.VideoFileClip(audio_path)
                if video.audio is None:
                    self.log("⚠️ 警告：源視頻沒有音訊軌道，將創建視頻但無法使用原視頻進行音頻替換")
            except Exception as e:
                self.log(f"⚠️ 檢查視頻時出錯: {str(e)}")
        
        # 禁用按鈕並顯示進度條
        self.process_btn.configure(state=tk.DISABLED)
        self.play_btn.configure(state=tk.DISABLED)
        self.save_btn.configure(state=tk.DISABLED)
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
            input_path = self.audio_path_var.get()
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
            
            # 確定要處理的音訊路徑
            audio_for_transcription = input_path
            if self.input_media_type == MEDIA_TYPES["VIDEO"] and self.extracted_audio_path:
                audio_for_transcription = self.extracted_audio_path
                self.log(f"🔄 使用從視頻中提取的音訊進行轉錄")
            
            # 轉錄音訊
            self.log(f"🎧 轉錄音訊中: {os.path.basename(audio_for_transcription)}")
            lang_config = LANGUAGE_PROMPTS[lang_mode]
            result = model.transcribe(audio_for_transcription, prompt=lang_config["prompt"], language=lang_config["language"])
            transcription = result['text']
            
            # 更新UI
            self.root.after(0, lambda: self.transcription_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.transcription_text.insert(tk.END, transcription))
            self.log(transcription)
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
            self.root.after(0, lambda: self.save_btn.configure(state=tk.NORMAL))
            
            # 如果有視頻輸入，則啟用視頻換臉按鈕
            if self.input_media_type == MEDIA_TYPES["VIDEO"]:
                self.root.after(0, lambda: self.retalk_btn.configure(state=tk.NORMAL))
                self.log("✅ 可以使用「視頻換臉」功能將翻譯後的音訊與原始視頻合成")
            
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
            self.root.after(0, lambda: self.play_btn.configure(state=tk.DISABLED))
            self.root.after(0, lambda: self.save_btn.configure(state=tk.DISABLED))
    
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
                    
                    # 獲取選定的輸出類型和格式
                    output_type = self.output_type_var.get().split(" - ")[0]
                    format_choice = self.format_var.get().split(" - ")[0]
                    
                    # 創建臨時目錄
                    output_temp_dir = tempfile.mkdtemp()
                    self.temp_files.append(output_temp_dir)
                    
                    # 首先保存為 WAV 格式（這是 XTTS 的原始輸出格式）
                    temp_wav_path = os.path.join(output_temp_dir, "output_temp.wav")
                    wav_write.write(temp_wav_path, sr, outputs["wav"])
                    
                    if output_type == "AUDIO":
                        # 處理音訊輸出
                        output_format = AUDIO_FORMATS[format_choice]["ext"]
                        final_output_path = f"output.{output_format}"
                        
                        if output_format == "wav":
                            # 如果是 WAV 格式，直接複製臨時文件
                            shutil.copy2(temp_wav_path, final_output_path)
                        else:
                            # 使用 pydub 轉換為其他格式
                            try:
                                audio = AudioSegment.from_wav(temp_wav_path)
                                
                                # 設置轉換參數
                                export_params = {}
                                if output_format == "mp3":
                                    export_params = {"bitrate": "192k"}
                                elif output_format == "m4a":
                                    export_params = {"bitrate": "192k", "format": "ipod"}
                                elif output_format == "ogg":
                                    export_params = {"bitrate": "192k"}
                                
                                # 導出為所選格式
                                audio.export(final_output_path, format=output_format, **export_params)
                                
                            except Exception as e:
                                self.log(f"❌ 格式轉換錯誤: {str(e)}")
                                # 如果轉換失敗，使用原始 WAV 文件作為備選
                                shutil.copy2(temp_wav_path, "output.wav")
                                final_output_path = "output.wav"
                        
                        self.log(f"✅ 成功保存音頻到 {final_output_path}")
                    
                    else:  # VIDEO 輸出
                        # 檢查是否有源視頻
                        if self.input_media_type != MEDIA_TYPES["VIDEO"]:
                            self.log("⚠️ 未找到源視頻，將使用音頻播放器外殼創建視頻")
                            # 創建無視頻的音頻視覺化（可選：將來可以擴展為波形或其他視覺效果）
                            self.create_audio_visual_video(temp_wav_path, format_choice)
                        else:
                            # 使用原始視頻替換音頻
                            input_video_path = self.audio_path_var.get()
                            self.create_video_with_new_audio(input_video_path, temp_wav_path, format_choice)
                    
                    # 設置當前輸出路徑，用於播放功能
                    self.current_output_path = final_output_path
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
    
    def create_audio_visual_video(self, audio_path, video_format):
        """從音頻創建簡單視頻（單色背景+音頻）"""
        try:
            self.log("🔄 正在創建音頻視覺化視頻...")
            
            # 獲取輸出格式
            output_format = VIDEO_FORMATS[video_format]["ext"]
            
            # 創建臨時目錄
            temp_dir = tempfile.mkdtemp()
            self.temp_files.append(temp_dir)
            
            # 使用 moviepy 創建視頻
            audio_clip = mp.AudioFileClip(audio_path)
            audio_duration = audio_clip.duration
            
            # 創建純色背景視頻（黑色背景）
            # 可以在這裡添加音頻可視化，例如波形等
            video_clip = mp.ColorClip(size=(1280, 720), color=(0, 0, 0), duration=audio_duration)
            
            # 添加標題文字
            txt_clip = mp.TextClip(
                "音訊語音合成 - 由多語言音訊處理器生成",
                fontsize=50, color='white', font="Arial-Bold", 
                size=(1000, 200)
            )
            txt_clip = txt_clip.set_position('center').set_duration(audio_duration)
            
            # 添加時間戳
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            time_txt = mp.TextClip(
                f"生成時間: {timestamp}",
                fontsize=30, color='white', font="Arial", 
                size=(800, 100)
            )
            time_txt = time_txt.set_position(('center', 500)).set_duration(audio_duration)
            
            # 組合視頻
            video_with_txt = mp.CompositeVideoClip([video_clip, txt_clip, time_txt])
            video_with_audio = video_with_txt.set_audio(audio_clip)
            
            # 保存視頻
            final_output_path = f"output.{output_format}"
            video_with_audio.write_videofile(
                final_output_path, 
                codec='libx264',
                audio_codec='aac', 
                temp_audiofile=os.path.join(temp_dir, "temp_audio.m4a"),
                remove_temp=True,
                fps=30
            )
            
            self.current_output_path = final_output_path
            self.log(f"✅ 成功生成視頻到 {final_output_path}")
            
        except Exception as e:
            self.log(f"❌ 創建視頻時出錯: {str(e)}")
            # 如果視頻生成失敗，回退到僅保存音頻
            output_format = "wav"
            shutil.copy2(audio_path, f"output.{output_format}")
            self.current_output_path = f"output.{output_format}"
            self.log(f"⚠️ 視頻創建失敗，已保存音頻到 {self.current_output_path}")
    
    def create_video_with_new_audio(self, video_path, audio_path, video_format):
        """使用原視頻但替換為新的音頻"""
        try:
            self.log("🔄 正在創建視頻（使用原視頻 + 新音頻）...")
            
            # 獲取輸出格式
            output_format = VIDEO_FORMATS[video_format]["ext"]
            
            # 創建臨時目錄
            temp_dir = tempfile.mkdtemp()
            self.temp_files.append(temp_dir)
            
            # 加載原視頻（但不使用其音頻）
            video_clip = mp.VideoFileClip(video_path)
            
            # 加載新音頻
            audio_clip = mp.AudioFileClip(audio_path)
            
            # 檢查音頻和視頻的時長，如果音頻較長，則延長視頻；如果視頻較長，則剪切視頻
            video_duration = video_clip.duration
            audio_duration = audio_clip.duration
            
            if audio_duration > video_duration:
                self.log(f"⚠️ 合成的音頻 ({audio_duration:.2f}秒) 比原視頻 ({video_duration:.2f}秒) 長，將重複視頻以匹配音頻長度")
                # 計算需要重複視頻的次數
                repeat_times = int(audio_duration / video_duration) + 1
                # 創建重複視頻
                repeated_clips = [video_clip] * repeat_times
                extended_clip = mp.concatenate_videoclips(repeated_clips)
                # 裁剪到音頻長度
                video_clip = extended_clip.subclip(0, audio_duration)
            elif video_duration > audio_duration:
                self.log(f"⚠️ 原視頻 ({video_duration:.2f}秒) 比合成的音頻 ({audio_duration:.2f}秒) 長，將裁剪視頻以匹配音頻長度")
                video_clip = video_clip.subclip(0, audio_duration)
            
            # 設置新音頻
            final_clip = video_clip.set_audio(audio_clip)
            
            # 保存視頻
            final_output_path = f"output.{output_format}"
            final_clip.write_videofile(
                final_output_path, 
                codec='libx264',
                audio_codec='aac', 
                temp_audiofile=os.path.join(temp_dir, "temp_audio.m4a"),
                remove_temp=True
            )
            
            self.current_output_path = final_output_path
            self.log(f"✅ 成功生成視頻到 {final_output_path}")
            
        except Exception as e:
            self.log(f"❌ 創建視頻時出錯: {str(e)}")
            # 如果視頻生成失敗，回退到僅保存音頻
            output_format = "wav"
            shutil.copy2(audio_path, f"output.{output_format}")
            self.current_output_path = f"output.{output_format}"
            self.log(f"⚠️ 視頻創建失敗，已保存音頻到 {self.current_output_path}")
    
    def play_output(self):
        """播放生成的音訊或視頻"""
        output_path = getattr(self, 'current_output_path', "output.wav")
        if not os.path.exists(output_path):
            self.log(f"❌ 找不到輸出檔案: {output_path}")
            return
            
        file_ext = os.path.splitext(output_path)[1].lower()
        
        # 檢查是否是視頻文件
        if file_ext in ['.mp4', '.mov', '.mkv']:
            self.log(f"🎬 正在嘗試播放視頻: {output_path}")
            try:
                # 使用系統默認程序打開視頻文件
                if os.name == 'nt':  # Windows
                    os.startfile(output_path)
                elif os.name == 'posix':  # macOS 和 Linux
                    if 'darwin' in os.sys.platform:  # macOS
                        subprocess.run(['open', output_path])
                    else:  # Linux
                        subprocess.run(['xdg-open', output_path])
                self.log("🎬 已使用系統視頻播放器打開視頻")
            except Exception as e:
                self.log(f"❌ 播放視頻時發生錯誤: {str(e)}")
        else:
            # 處理音訊文件播放
            try:
                # 檢查是否是支援的格式
                file_ext = os.path.splitext(output_path)[1].lower()
                
                # 如果不是 WAV 格式，需要先轉換為臨時 WAV 文件才能播放
                temp_wav_for_play = None
                
                if file_ext != '.wav':
                    try:
                        self.log(f"🔄 正在準備播放 {file_ext} 格式音訊...")
                        # 使用 pydub 加載音訊並轉為臨時的 WAV 文件
                        audio = None
                        
                        if file_ext == '.mp3':
                            audio = AudioSegment.from_mp3(output_path)
                        elif file_ext == '.m4a':
                            audio = AudioSegment.from_file(output_path, format="m4a")
                        elif file_ext == '.ogg':
                            audio = AudioSegment.from_ogg(output_path)
                        else:
                            audio = AudioSegment.from_file(output_path)
                        
                        if audio:
                            temp_wav_for_play = "play_temp.wav"
                            audio.export(temp_wav_for_play, format="wav")
                            # 使用臨時文件進行播放
                            pygame.mixer.music.load(temp_wav_for_play)
                        else:
                            raise Exception("無法加載音訊格式")
                    except Exception as e:
                        self.log(f"❌ 準備播放時出錯: {str(e)}，嘗試直接播放原始文件")
                        # 如果轉換失敗，嘗試直接播放
                        pygame.mixer.music.load(output_path)
                else:
                    # WAV 格式直接播放
                    pygame.mixer.music.load(output_path)
                
                # 播放音訊
                pygame.mixer.music.play()
                self.log("🎵 正在播放合成的音訊...")
                
                # 播放完成後，刪除臨時文件
                def cleanup_temp_file():
                    pygame.time.wait(int(pygame.mixer.music.get_length() * 1000) + 500)  # 等待播放完成
                    if temp_wav_for_play and os.path.exists(temp_wav_for_play):
                        try:
                            os.remove(temp_wav_for_play)
                        except:
                            pass
                
                # 創建一個清理臨時文件的線程
                if temp_wav_for_play:
                    cleanup_thread = threading.Thread(target=cleanup_temp_file)
                    cleanup_thread.daemon = True
                    cleanup_thread.start()
                
            except Exception as e:
                self.log(f"❌ 播放音訊時發生錯誤: {str(e)}")

    def check_ffmpeg(self):
        """檢查系統是否安裝了 FFmpeg，這對某些音訊格式轉換是必需的"""
        try:
            # 嘗試執行 ffmpeg 命令來檢查它是否可用
            result = subprocess.run(['ffmpeg', '-version'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    check=False)
            
            if result.returncode == 0:
                ffmpeg_version = result.stdout.split('\n')[0]
                self.log(f"✅ 檢測到 FFmpeg: {ffmpeg_version}")
                return True
            else:
                self.log("⚠️ 未檢測到 FFmpeg，某些格式轉換可能無法工作")
                return False
        except Exception as e:
            self.log(f"⚠️ FFmpeg 檢測失敗: {str(e)}")
            return False
    
    def save_as(self):
        """允許用戶將音訊或視頻另存為不同的格式和位置"""
        if not hasattr(self, 'current_output_path') or not os.path.exists(self.current_output_path):
            self.log("❌ 沒有可用的輸出檔案")
            return
        
        # 獲取當前輸出檔案的副檔名
        current_ext = os.path.splitext(self.current_output_path)[1].lower()
        is_video = current_ext in ['.mp4', '.mov', '.mkv']
        
        # 定義可用格式的過濾器
        if is_video:
            file_types = [
                ("MP4 視頻", "*.mp4"),
                ("MOV 視頻", "*.mov"),
                ("MKV 視頻", "*.mkv"),
                ("所有檔案", "*.*")
            ]
        else:
            file_types = [
                ("WAV 音訊", "*.wav"),
                ("MP3 音訊", "*.mp3"),
                ("M4A 音訊", "*.m4a"),
                ("OGG 音訊", "*.ogg"),
                ("所有檔案", "*.*")
            ]
        
        # 獲取當前輸出檔案的基本名稱
        current_name = os.path.basename(self.current_output_path)
        current_name_no_ext = os.path.splitext(current_name)[0]
        
        # 打開保存對話框
        save_path = filedialog.asksaveasfilename(
            title="保存檔案",
            filetypes=file_types,
            defaultextension=current_ext,
            initialfile=current_name_no_ext
        )
        
        if save_path:
            try:
                self.log(f"🔄 正在保存檔案到 {save_path}...")
                
                # 獲取目標格式
                target_ext = os.path.splitext(save_path)[1].lower()
                if not target_ext:
                    target_ext = current_ext  # 使用當前格式作為默認值
                    save_path += target_ext
                
                # 獲取源文件格式
                source_ext = os.path.splitext(self.current_output_path)[1].lower()
                
                # 檢查是否在視頻和音訊之間轉換（不支持）
                if (is_video and target_ext not in ['.mp4', '.mov', '.mkv']) or \
                   (not is_video and target_ext in ['.mp4', '.mov', '.mkv']):
                    self.log("❌ 不支持在視頻和音訊格式之間直接轉換，請使用處理功能")
                    return
                
                # 如果源文件和目標格式相同，直接複製
                if source_ext == target_ext:
                    shutil.copy2(self.current_output_path, save_path)
                    self.log(f"✅ 已保存檔案到 {save_path}")
                    return
                
                # 需要進行格式轉換
                try:
                    if is_video:
                        # 視頻轉換
                        self.convert_video_format(self.current_output_path, save_path)
                    else:
                        # 音訊轉換
                        self.convert_audio_format(self.current_output_path, save_path)
                
                except Exception as e:
                    self.log(f"❌ 格式轉換失敗: {str(e)}")
                    # 如果轉換失敗，嘗試直接複製原始文件
                    shutil.copy2(self.current_output_path, save_path)
                    self.log(f"⚠️ 已直接複製原始檔案到: {save_path}")
            
            except Exception as e:
                self.log(f"❌ 保存檔案時發生錯誤: {str(e)}")
    
    def convert_video_format(self, source_path, target_path):
        """轉換視頻格式"""
        try:
            # 創建臨時目錄
            temp_dir = tempfile.mkdtemp()
            self.temp_files.append(temp_dir)
            
            # 加載視頻
            video = mp.VideoFileClip(source_path)
            
            # 獲取目標格式
            target_ext = os.path.splitext(target_path)[1].lower()[1:]  # 移除點號
            
            # 設置不同格式的參數
            if target_ext == 'mp4':
                video.write_videofile(
                    target_path, 
                    codec='libx264',
                    audio_codec='aac', 
                    temp_audiofile=os.path.join(temp_dir, "temp_audio.m4a"),
                    remove_temp=True
                )
            elif target_ext == 'mov':
                video.write_videofile(
                    target_path, 
                    codec='libx264',
                    audio_codec='aac', 
                    temp_audiofile=os.path.join(temp_dir, "temp_audio.m4a"),
                    remove_temp=True
                )
            elif target_ext == 'mkv':
                video.write_videofile(
                    target_path, 
                    codec='libx264',
                    audio_codec='libvorbis', 
                    temp_audiofile=os.path.join(temp_dir, "temp_audio.ogg"),
                    remove_temp=True
                )
            
            self.log(f"✅ 已成功將視頻保存為 {target_ext.upper()} 格式: {target_path}")
            
        except Exception as e:
            self.log(f"❌ 視頻格式轉換失敗: {str(e)}")
            raise e
    
    def convert_audio_format(self, source_path, target_path):
        """轉換音訊格式"""
        # 使用 pydub 加載音訊
        source_ext = os.path.splitext(source_path)[1].lower()
        target_ext = os.path.splitext(target_path)[1].lower()
        target_format = target_ext[1:]  # 去掉點號
        
        if source_ext == '.wav':
            audio = AudioSegment.from_wav(source_path)
        elif source_ext == '.mp3':
            audio = AudioSegment.from_mp3(source_path)
        elif source_ext == '.m4a':
            audio = AudioSegment.from_file(source_path, format="m4a")
        elif source_ext == '.ogg':
            audio = AudioSegment.from_ogg(source_path)
        else:
            audio = AudioSegment.from_file(source_path)
        
        # 設置轉換參數
        export_params = {}
        if target_format == "mp3":
            export_params = {"bitrate": "192k"}
        elif target_format == "m4a":
            export_params = {"bitrate": "192k", "format": "ipod"}
        elif target_format == "ogg":
            export_params = {"bitrate": "192k"}
        
        # 導出為目標格式
        audio.export(target_path, format=target_format, **export_params)
        self.log(f"✅ 已成功將音訊保存為 {target_format.upper()} 格式: {target_path}")
    
    def start_video_retalk(self):
        """啟動視頻換臉處理對話框並執行"""
        # 檢查輸出音訊是否存在
        if not hasattr(self, 'current_output_path') or not os.path.exists(self.current_output_path):
            self.log("❌ 沒有可用的輸出音訊檔案，請先處理音訊")
            messagebox.showerror("錯誤", "沒有可用的輸出音訊檔案，請先處理音訊")
            return
            
        # 檢查是否有視頻retalking目錄
        if not os.path.exists("video-retalking"):
            self.log("❌ 找不到 video-retalking 目錄，請確保已安裝")
            messagebox.showerror("錯誤", "找不到 video-retalking 目錄，請確保已安裝")
            return
            
        # 創建對話框窗口
        retalk_dialog = tk.Toplevel(self.root)
        retalk_dialog.title("視頻換臉設置")
        retalk_dialog.geometry("600x400")
        retalk_dialog.grab_set()  # 使對話框成為模態
        
        # 創建主框架
        main_frame = ttk.Frame(retalk_dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 臉部視頻選擇區域
        face_frame = ttk.LabelFrame(main_frame, text="臉部視頻", padding=10)
        face_frame.pack(fill=tk.X, pady=5)
        
        face_path_var = tk.StringVar()
        ttk.Entry(face_frame, textvariable=face_path_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_face_video():
            file_path = filedialog.askopenfilename(
                title="選擇包含臉部的視頻",
                filetypes=[
                    ("視頻檔案", "*.mp4 *.mov *.mkv"),
                    ("所有檔案", "*.*")
                ]
            )
            if file_path:
                face_path_var.set(file_path)
        
        ttk.Button(face_frame, text="瀏覽", command=browse_face_video).pack(side=tk.LEFT, padx=5)
        
        # 音訊選擇區域
        audio_frame = ttk.LabelFrame(main_frame, text="音訊文件", padding=10)
        audio_frame.pack(fill=tk.X, pady=5)
        
        audio_path_var = tk.StringVar(value=self.current_output_path)
        ttk.Entry(audio_frame, textvariable=audio_path_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_audio_file():
            file_path = filedialog.askopenfilename(
                title="選擇音訊檔案",
                filetypes=[
                    ("音訊檔案", "*.wav *.mp3 *.ogg *.m4a"),
                    ("所有檔案", "*.*")
                ]
            )
            if file_path:
                audio_path_var.set(file_path)
        
        ttk.Button(audio_frame, text="瀏覽", command=browse_audio_file).pack(side=tk.LEFT, padx=5)
        
        # 輸出設置區域
        output_frame = ttk.LabelFrame(main_frame, text="輸出設置", padding=10)
        output_frame.pack(fill=tk.X, pady=5)
        
        output_path_var = tk.StringVar()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = f"video-retalking/results/output_{timestamp}.mp4"
        output_path_var.set(default_output)
        
        ttk.Entry(output_frame, textvariable=output_path_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_output_file():
            file_path = filedialog.asksaveasfilename(
                title="設置輸出文件",
                filetypes=[("MP4 視頻", "*.mp4")],
                defaultextension=".mp4",
                initialfile=f"output_{timestamp}.mp4"
            )
            if file_path:
                output_path_var.set(file_path)
        
        ttk.Button(output_frame, text="瀏覽", command=browse_output_file).pack(side=tk.LEFT, padx=5)
        
        # 進度和狀態區域
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=10)
        
        status_var = tk.StringVar(value="準備就緒")
        status_label = ttk.Label(status_frame, textvariable=status_var)
        status_label.pack(side=tk.LEFT, padx=5)
        
        progress = ttk.Progressbar(status_frame, mode='indeterminate')
        progress.pack(fill=tk.X, expand=True, padx=5, pady=5)
        
        # 按鈕區域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def on_start():
            # 檢查必要的輸入
            face_path = face_path_var.get()
            audio_path = audio_path_var.get()
            output_path = output_path_var.get()
            
            if not face_path:
                messagebox.showerror("錯誤", "請選擇臉部視頻")
                return
                
            if not audio_path:
                messagebox.showerror("錯誤", "請選擇音訊檔案")
                return
            
            if not os.path.exists(face_path):
                messagebox.showerror("錯誤", f"臉部視頻不存在: {face_path}")
                return
                
            if not os.path.exists(audio_path):
                messagebox.showerror("錯誤", f"音訊檔案不存在: {audio_path}")
                return
            
            # 禁用按鈕，顯示進度
            start_btn.configure(state=tk.DISABLED)
            cancel_btn.configure(state=tk.DISABLED)
            progress.start()
            status_var.set("處理中...")
            
            # 在新線程中運行處理以避免凍結UI
            def process_thread():
                try:
                    result = self.video_retalk(face_path, audio_path, output_path)
                    
                    # 在主線程中更新UI
                    self.root.after(0, lambda: progress.stop())
                    
                    if result:
                        self.root.after(0, lambda: status_var.set("處理完成"))
                        self.root.after(0, lambda: messagebox.showinfo("成功", f"視頻換臉處理成功!\n輸出檔案: {result}"))
                        self.root.after(0, lambda: retalk_dialog.destroy())
                        
                        # 更新主界面的當前輸出路徑並啟用播放按鈕
                        self.current_output_path = result
                        self.root.after(0, lambda: self.play_btn.configure(state=tk.NORMAL))
                        self.root.after(0, lambda: self.save_btn.configure(state=tk.NORMAL))
                    else:
                        self.root.after(0, lambda: status_var.set("處理失敗"))
                        self.root.after(0, lambda: messagebox.showerror("錯誤", "視頻換臉處理失敗"))
                        self.root.after(0, lambda: start_btn.configure(state=tk.NORMAL))
                        self.root.after(0, lambda: cancel_btn.configure(state=tk.NORMAL))
                
                except Exception as e:
                    self.root.after(0, lambda: progress.stop())
                    self.root.after(0, lambda: status_var.set("處理錯誤"))
                    self.root.after(0, lambda: messagebox.showerror("錯誤", f"處理時發生錯誤: {str(e)}"))
                    self.root.after(0, lambda: start_btn.configure(state=tk.NORMAL))
                    self.root.after(0, lambda: cancel_btn.configure(state=tk.NORMAL))
            
            # 啟動處理線程
            threading.Thread(target=process_thread, daemon=True).start()
        
        def on_cancel():
            retalk_dialog.destroy()
        
        start_btn = ttk.Button(button_frame, text="開始處理", command=on_start)
        start_btn.pack(side=tk.RIGHT, padx=5)
        
        cancel_btn = ttk.Button(button_frame, text="取消", command=on_cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        # 如果目前有輸入的視頻，自動填入為臉部視頻
        if self.input_media_type == MEDIA_TYPES["VIDEO"]:
            face_path_var.set(self.audio_path_var.get())
    
    def video_retalk(self, face_video_path, audio_path, output_path=None):
        """使用 video-retalking 技術將音訊同步到臉部視頻"""
        try:
            self.log("🎬 正在啟動視頻換聲技術處理...")
            
            # 設定環境變數
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            # 如果未提供輸出路徑，則生成一個基於時間戳的路徑
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"video-retalking/results/output_{timestamp}.mp4"
            
            # 確保輸出目錄存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 設定 video-retalking 資料夾為工作目錄
            video_retalking_dir = "video-retalking"
            
            # 執行 video-retalking 處理
            self.log(f"🔄 正在處理視頻，這可能需要一些時間...")
            result = subprocess.run([
                "python3", "inference.py",
                "--face", face_video_path,
                "--audio", audio_path,
                "--outfile", output_path
            ], capture_output=True, text=True, cwd=video_retalking_dir)  # 設定工作目錄
            
            # 檢查是否成功
            if result.returncode == 0:
                self.log(f"✅ 視頻換聲處理成功，輸出檔案: {output_path}")
                return output_path
            else:
                self.log(f"❌ 視頻換聲處理失敗: {result.stderr}")
                return None
        
        except Exception as e:
            self.log(f"❌ 視頻換聲處理時發生錯誤: {str(e)}")
            return None


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioProcessorApp(root)
    
    # 在程序結束時清理臨時文件
    def on_closing():
        # 關閉所有可能的子窗口
        for widget in root.winfo_children():
            if isinstance(widget, tk.Toplevel):
                widget.destroy()
                
        app.cleanup_temp_files()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
        