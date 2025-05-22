import tkinter as tk
import sounddevice as sd
import numpy as np
import librosa

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite  # fallback nếu test trên máy khác

# === Configuration ===
SAMPLE_RATE = 16000
DURATION = 1  # seconds
SAMPLES = SAMPLE_RATE * DURATION
N_MELS = 40
KEYWORDS = ["bed", "bird", "cat", "dog", "house"]
MODEL_PATH = "model/keyword_model.tflite"

# === Load TFLite Model ===
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Audio and Prediction ===
def record_audio():
    try:
        audio = sd.rec(SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        return audio.flatten()
    except Exception as e:
        result_var.set(f"🎤 Mic error: {str(e)}")
        return np.zeros(SAMPLES)

def extract_features(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def predict_from_mic():
    try:
        result_var.set("🎧 Listening...")
        root.update_idletasks()

        audio = record_audio()
        features = extract_features(audio)
        input_data = features[np.newaxis, ..., np.newaxis].astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_idx = int(np.argmax(output_data))
        confidence = float(output_data[predicted_idx])

        result_var.set(f"🔊 Detected: '{KEYWORDS[predicted_idx]}' ({confidence:.2%} confidence)")
    except Exception as e:
        result_var.set(f"❌ Error: {str(e)}")

# === GUI Setup ===
root = tk.Tk()
root.title("🎤 Keyword Recognition (Raspberry Pi)")
root.geometry("500x250")
root.configure(bg="white")

title_label = tk.Label(
    root, text="🎙 Keyword Spotting", font=("Arial", 24, "bold"),
    bg="white", fg="#2c3e50"
)
title_label.pack(pady=(20, 10))

result_var = tk.StringVar(value="Press 'Speak' to start")
result_label = tk.Label(
    root, textvariable=result_var, font=("Arial", 16), bg="#ecf0f1",
    fg="#2d3436", width=40, height=2, relief="groove", bd=2
)
result_label.pack(pady=20)

speak_button = tk.Button(
    root, text="🎙 SPEAK", font=("Arial", 18, "bold"),
    bg="#2980b9", fg="white", activebackground="#3498db",
    command=predict_from_mic, width=20, height=2
)
speak_button.pack(pady=10)

footer_label = tk.Label(
    root, text="Running on Raspberry Pi", font=("Arial", 10),
    bg="white", fg="#95a5a6"
)
footer_label.pack(side="bottom", pady=10)

root.mainloop()
