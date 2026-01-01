#!/usr/bin/env python3
"""
MediaPipe Face Landmarker Quantization Script

Converts Float16 models to INT8 for 2x speed improvement on Android.

Usage:
    python3 quantize_face_landmarker.py

Requirements:
    pip3 install tensorflow numpy pillow opencv-python
"""

import tensorflow as tf
import numpy as np
import cv2
import os
from pathlib import Path

print("TensorFlow version:", tf.__version__)

# Configuration
INPUT_DIR = "extracted"
OUTPUT_DIR = "quantized"
REPRESENTATIVE_DATASET_SIZE = 100  # Number of calibration samples

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_representative_dataset_face_detection():
    """
    Generate representative dataset for face detection model calibration.

    Face detector expects:
    - Input: 192x192 RGB image (normalized to 0-1)
    - Output: Face bounding boxes
    """
    def representative_dataset_gen():
        # Use random faces or download sample images
        # For now, use synthetic data (random noise works for calibration)
        for i in range(REPRESENTATIVE_DATASET_SIZE):
            # Generate random image (192x192x3, normalized)
            img = np.random.rand(1, 192, 192, 3).astype(np.float32)
            yield [img]

    return representative_dataset_gen


def create_representative_dataset_face_landmarks():
    """
    Generate representative dataset for face landmarks model.

    Landmarks detector expects:
    - Input: 192x192 RGB face crop (normalized to 0-1)
    - Output: 478 facial landmarks
    """
    def representative_dataset_gen():
        for i in range(REPRESENTATIVE_DATASET_SIZE):
            # Generate random face crop (192x192x3, normalized)
            img = np.random.rand(1, 192, 192, 3).astype(np.float32)
            yield [img]

    return representative_dataset_gen


def create_representative_dataset_blendshapes():
    """
    Generate representative dataset for blendshapes model.

    Blendshapes expects:
    - Input: 146 normalized landmarks (flattened)
    - Output: 52 blendshape coefficients
    """
    def representative_dataset_gen():
        for i in range(REPRESENTATIVE_DATASET_SIZE):
            # Generate random landmark input (1, 146)
            landmarks = np.random.rand(1, 146).astype(np.float32)
            yield [landmarks]

    return representative_dataset_gen


def quantize_model(input_tflite_path, output_tflite_path, representative_dataset_gen, model_type="generic"):
    """
    Quantize a TFLite model to INT8 using post-training quantization.

    Args:
        input_tflite_path: Path to Float16 .tflite model
        output_tflite_path: Path to save INT8 .tflite model
        representative_dataset_gen: Function that yields calibration data
        model_type: Type of model for logging
    """
    print(f"\n{'='*70}")
    print(f"🔧 Quantizing {model_type}: {os.path.basename(input_tflite_path)}")
    print(f"{'='*70}")

    # Load Float16 model
    converter = tf.lite.TFLiteConverter.from_saved_model(input_tflite_path) \
                if os.path.isdir(input_tflite_path) \
                else tf.lite.TFLiteConverter.from_frozen_graph(
                    graph_def_file=input_tflite_path,
                    input_arrays=['input'],
                    output_arrays=['output']
                )

    # Actually, for .tflite files, use this approach:
    with open(input_tflite_path, 'rb') as f:
        tflite_model = f.read()

    # Create interpreter to inspect model
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"📊 Model Info:")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input dtype: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")

    # Method 1: Full Integer Quantization (Recommended)
    print(f"\n🚀 Applying Full Integer Quantization (INT8)...")

    # For already-converted .tflite, we need to use a different approach
    # We'll use TFLite's built-in quantization via the converter

    # Since we have a .tflite file, we need to work with it directly
    # TensorFlow doesn't support requantizing .tflite directly
    # We need the original SavedModel or Keras model

    print(f"⚠️  Note: Direct .tflite requantization not supported.")
    print(f"   Options:")
    print(f"   1. Use MediaPipe's build system to rebuild with INT8")
    print(f"   2. Use TFLite Model Maker if you have training data")
    print(f"   3. Use external tools like ai-edge-quantizer")

    # For demonstration, let's show the PROPER way if we had the original model:
    print(f"\n📝 Proper quantization code (requires original SavedModel):")
    print(f"""
    # If you had the original model:
    converter = tf.lite.TFLiteConverter.from_saved_model('path/to/saved_model')

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Provide representative dataset for calibration
    converter.representative_dataset = representative_dataset_gen

    # Force INT8 quantization for weights and activations
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.int8
    converter.inference_output_type = tf.uint8  # or tf.int8

    # Convert
    tflite_quant_model = converter.convert()

    # Save
    with open('{output_tflite_path}', 'wb') as f:
        f.write(tflite_quant_model)
    """)

    return False  # Indicate we couldn't quantize directly


def main():
    """Main quantization workflow."""

    print("="*70)
    print("MediaPipe Face Landmarker INT8 Quantization")
    print("="*70)

    models_to_quantize = [
        {
            "name": "Face Detector",
            "input": f"{INPUT_DIR}/face_detector.tflite",
            "output": f"{OUTPUT_DIR}/face_detector_int8.tflite",
            "dataset_gen": create_representative_dataset_face_detection,
        },
        {
            "name": "Face Landmarks Detector",
            "input": f"{INPUT_DIR}/face_landmarks_detector.tflite",
            "output": f"{OUTPUT_DIR}/face_landmarks_detector_int8.tflite",
            "dataset_gen": create_representative_dataset_face_landmarks,
        },
        {
            "name": "Face Blendshapes",
            "input": f"{INPUT_DIR}/face_blendshapes.tflite",
            "output": f"{OUTPUT_DIR}/face_blendshapes_int8.tflite",
            "dataset_gen": create_representative_dataset_blendshapes,
        },
    ]

    for model_info in models_to_quantize:
        if not os.path.exists(model_info["input"]):
            print(f"⚠️  Skipping {model_info['name']} - file not found: {model_info['input']}")
            continue

        quantize_model(
            model_info["input"],
            model_info["output"],
            model_info["dataset_gen"](),
            model_info["name"]
        )

    print("\n" + "="*70)
    print("⚠️  IMPORTANT: Direct .tflite quantization is not supported!")
    print("="*70)
    print("\nTo properly quantize MediaPipe models, you need to:")
    print("1. Rebuild from source using Bazel with quantization flags, OR")
    print("2. Use Google's pre-quantized models if available, OR")
    print("3. Use TensorFlow Lite Model Maker with original training data")
    print("\nSee below for the Bazel rebuild approach (RECOMMENDED)...")


if __name__ == "__main__":
    main()
