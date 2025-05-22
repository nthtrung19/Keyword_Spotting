#!/usr/bin/env python3
"""
convert_to_tflite.py – Convert a Keras .h5 model to a TFLite model while preserving accuracy.

Usage:
    python convert_to_tflite.py \
      --h5 model/keyword_spotting_custom_final.h5 \
      --out model/keyword_spotting_fp32.tflite \
      [--mode float32|dynamic|float16] \
      [--reproot test]

Options:
    --h5        Input Keras .h5 file (required)
    --out       Output TFLite file path (required)
    --mode      Conversion mode, default float32: no quantization; dynamic: weight quantization; float16: weight float16 quantization
    --reproot   Directory of WAVs for representative data (for dynamic/float16 modes)

Requirements:
    pip install tensorflow numpy soundfile librosa
"""
import argparse
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf

# Preprocessing constants must match inference pipeline
SR = 16000
DURATION = 0.7
SAMPLES = int(SR * DURATION)
N_MELS = 70
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 320
FRAMES = 69

# Generator for representative dataset
def representative_data_gen(wav_dir, num_samples=100):
    import glob
    for path in glob.glob(f"{wav_dir}/**/*.wav", recursive=True)[:num_samples]:
        wav, sr = sf.read(path, dtype='float32')
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        if len(wav) < SAMPLES:
            wav = np.pad(wav, (0, SAMPLES - len(wav)))
        else:
            wav = wav[:SAMPLES]
        m = librosa.feature.melspectrogram(y=wav, sr=SR,
            n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_mels=N_MELS)
        log_mel = np.log1p(m)
        log_mel = (log_mel - log_mel.mean())/(log_mel.std()+1e-6)
        spec = log_mel.T
        if spec.shape[0] < FRAMES:
            spec = np.pad(spec, ((0, FRAMES - spec.shape[0]), (0, 0)))
        yield [spec[np.newaxis, :, :, np.newaxis].astype(np.float32)]

# Main conversion
def main():
    parser = argparse.ArgumentParser(description="Convert .h5 to TFLite with options")
    parser.add_argument('--h5', required=True, help='Input Keras .h5 model')
    parser.add_argument('--out', required=True, help='Output TFLite path')
    parser.add_argument('--mode', choices=['float32','dynamic','float16'], default='float32', help='Conversion mode')
    parser.add_argument('--reproot', help='Representative WAV folder', default=None)
    args = parser.parse_args()

    # Load Keras model
    print(f"Loading Keras model from {args.h5}...")
    model = tf.keras.models.load_model(args.h5)

    # Set up converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.mode == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if args.reproot:
            converter.representative_dataset = lambda: representative_data_gen(args.reproot)
    elif args.mode == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        if args.reproot:
            converter.representative_dataset = lambda: representative_data_gen(args.reproot)
    # float32: no quantization

    # Convert
    print(f"Converting to TFLite ({args.mode})...")
    tflite_model = converter.convert()

    # Save
    with open(args.out, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {args.out}")

if __name__ == '__main__':
    main()
