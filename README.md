# 🎤 Voice Keyword Spotting (5 Words)

This project implements a real-time keyword spotting system that recognizes 5 specific spoken words: bed, bird, cat, dog, and house. The model is trained on Google Colab using log-mel spectrogram features and a lightweight CNN, then exported as a TensorFlow Lite model for deployment on both PC and Raspberry Pi.

## 🧠 Model Training (Google Colab)

Training is done in `Detect_keyword.ipynb`. Audio data (.wav, mono, 16kHz, 1 second) is loaded from subfolders named by keyword. Features are extracted using librosa's log-mel spectrogram. The model is a 2-layer Conv2D CNN followed by Dense layers and a softmax output. The trained model is converted to TFLite format using `tf.lite.TFLiteConverter` and saved as `keyword_model.tflite`.

## 💻 Run on Local PC

Run `python keyword_spotting_pc.py`. This script records a 1-second audio sample from your microphone, converts it to log-mel spectrogram, performs inference using the TFLite model, and shows the predicted keyword in a Tkinter GUI.

Dependencies: tensorflow, librosa, sounddevice.

## 🍓 Run on Raspberry Pi

Run `python3 keyword_spotting_rpi.py`. This version uses `tflite_runtime` for lightweight inference and works with USB or onboard microphones. GUI is built with Tkinter and optimized for RPi.

Install with:  
sudo apt install libportaudio2  
pip3 install tflite-runtime librosa sounddevice numpy

## 📁 Project Structure

Voice_Recognition/  
├── Detect_keyword.ipynb ← training notebook on Google Colab  
├── keyword_spotting_pc.py ← desktop test GUI using mic  
├── keyword_spotting_rpi.py ← Raspberry Pi inference  
├── model/  
│   └── keyword_model.tflite ← final TFLite model  

## 📌 Keywords Recognized

bed, bird, cat, dog, house


Made with 💡 for real-time embedded AI.
