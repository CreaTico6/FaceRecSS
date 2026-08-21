#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download OpenCV Zoo face detection (YuNet) + recognition (SFace) models."""

import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def download():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            print(f"  ✓ {name} já existe")
            continue
        print(f"  ⬇ A descarregar {name}...")
        try:
            urllib.request.urlretrieve(url, path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  ✓ {name} descarregado ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ✗ Erro ao descarregar {name}: {e}", file=sys.stderr)
            # Remove partial download
            if os.path.exists(path):
                os.remove(path)
            sys.exit(1)

    # Clean up old models (from previous Caffe-based approach)
    old_files = [
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel",
        "openface_nn4.small2.v1.t7",
    ]
    for old in old_files:
        old_path = os.path.join(MODELS_DIR, old)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f"  🗑 Removido modelo antigo: {old}")

    print("\nTodos os modelos prontos em:", MODELS_DIR)


if __name__ == "__main__":
    print("=== Download de Modelos para Reconhecimento Facial ===\n")
    download()
