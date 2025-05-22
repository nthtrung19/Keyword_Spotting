#!/usr/bin/env python3

import os
import sys
import warnings
import collections
import time
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox

# ==== Cấu hình âm thanh & model ====
MODEL_PATH    = "model/test.tflite"
SR            = 16000
WINDOW_SEC    = 0.7
STEP_SEC      = 0.1
WINDOW_SAMPS  = int(WINDOW_SEC * SR)
STEP_SAMPS    = int(STEP_SEC * SR)
N_MELS        = 70
N_FFT         = 512
HOP_LENGTH    = 160
WIN_LENGTH    = 320
FRAMES        = 69
LABELS        = ['bed', 'cake', 'phone']
RMS_THRESHOLD = 0.03

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class SimpleKWSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Keyword Spotting")
        self.geometry("600x400")
        self.configure(bg="white")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Load model
        self.interp, self.inp, self.out = self.load_model(MODEL_PATH)

        # State
        self.ring = collections.deque(maxlen=WINDOW_SAMPS)
        self.last_label = None
        self.last_detect_time = 0.0
        self.cooldown_sec = 3.0
        self.stream = None

        self.create_ui()

    def load_model(self, path):
        if not os.path.exists(path):
            messagebox.showerror("Model not found", path)
            sys.exit(1)
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        return interp, inp, out

    def create_ui(self):
        self.speak_btn = tk.Button(
            self, text="🎙 SPEAK", font=("Arial", 28), bg="#3498db", fg="white",
            command=self.start_stream, state="normal", width=12, height=2
        )
        self.speak_btn.pack(pady=40)

        self.product_label_var = tk.StringVar(value="PRODUCT")
        self.product_label = tk.Label(
            self, textvariable=self.product_label_var,
            font=("Arial", 28), fg="black", bg="#ecf0f1", width=20, height=2
        )
        self.product_label.pack(pady=20)

    def start_stream(self):
        self.speak_btn["state"] = "disabled"
        self.product_label_var.set("Listening...")
        self.ring.clear()
        self.last_label = None
        self.last_detect_time = 0.0

        self.stream = sd.InputStream(
            channels=1,
            samplerate=SR,
            blocksize=STEP_SAMPS,
            callback=self.audio_callback
        )
        self.stream.start()

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def audio_callback(self, indata, frames, time_info, status):
        if status.input_overflow:
            return
        mono = indata[:, 0]
        self.ring.extend(mono)
        if len(self.ring) < WINDOW_SAMPS:
            return

        buf = np.array(self.ring)
        rms = np.sqrt(np.mean(buf ** 2))
        if rms < RMS_THRESHOLD:
            return

        idx, scores = self.infer(buf)
        label = LABELS[idx]
        confidence = scores[idx]
        now = time.time()

        if (now - self.last_detect_time) >= self.cooldown_sec:
            self.last_detect_time = now
            self.after(0, lambda: self.show_result(label, confidence))

    def infer(self, audio):
        spec = librosa.feature.melspectrogram(
            y=audio, sr=SR, n_fft=N_FFT,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            n_mels=N_MELS, center=False
        )
        log_mel = np.log1p(spec)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        spec = log_mel.T
        if spec.shape[0] < FRAMES:
            spec = np.pad(spec, ((0, FRAMES - spec.shape[0]), (0, 0)))
        spec = spec[:FRAMES]
        inp_tensor = spec[np.newaxis, ..., np.newaxis].astype(np.float32)
        self.interp.set_tensor(self.inp['index'], inp_tensor)
        self.interp.invoke()
        return int(np.argmax(self.interp.get_tensor(self.out['index'])[0])), self.interp.get_tensor(self.out['index'])[0]

    def show_result(self, label, conf):
        self.product_label_var.set(f"{label.upper()} ({conf:.2f})")
        self.stop_stream()
        self.speak_btn["state"] = "normal"

    def on_close(self):
        self.stop_stream()
        self.destroy()


if __name__ == "__main__":
    app = SimpleKWSApp()
    app.mainloop()
