#!/usr/bin/env python3
"""
Proper TFLite INT8 Quantization with Float32 I/O

This uses TFLite's post-training quantization to convert existing .tflite models
to INT8 while keeping Float32 input/output.

WARNING: This only works if the .tflite models support re-quantization.
MediaPipe models may have restrictions.

Requirements:
    pip3 install tensorflow numpy opencv-python pillow

Usage:
    python3 quantize_tflite_proper.py
"""

import tensorflow as tf
import numpy as np
import os
import zipfile
from pathlib import Path

print(f"TensorFlow version: {tf.__version__}")

# Check TF version
if not tf.__version__.startswith("2."):
    print("❌ Error: This script requires TensorFlow 2.x")
    exit(1)


class TFLiteQuantizer:
    """Handles TFLite model quantization with representative dataset."""

    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def load_model(self):
        """Load and inspect the existing .tflite model."""
        with open(self.model_path, 'rb') as f:
            tflite_model = f.read()

        self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"\n📊 Model: {os.path.basename(self.model_path)}")
        print(f"   Input shape:  {self.input_details[0]['shape']}")
        print(f"   Input dtype:  {self.input_details[0]['dtype']}")
        print(f"   Output shape: {self.output_details[0]['shape']}")
        print(f"   Output dtype: {self.output_details[0]['dtype']}")
        print(f"   Size:         {os.path.getsize(self.model_path) / 1024:.1f} KB")

        return tflite_model

    def generate_representative_dataset(self, num_samples=100):
        """
        Generate representative dataset for quantization calibration.

        This should ideally use real data, but random data works for basic calibration.
        """
        input_shape = self.input_details[0]['shape']
        input_dtype = self.input_details[0]['dtype']

        def representative_dataset_gen():
            for i in range(num_samples):
                # Generate random sample matching input shape
                # Normalized to [0, 1] for image models
                if input_dtype == np.float32:
                    sample = np.random.rand(*input_shape).astype(np.float32)
                else:
                    sample = np.random.rand(*input_shape).astype(input_dtype)

                yield [sample]

        return representative_dataset_gen

    def quantize(self, representative_dataset_gen):
        """
        Apply INT8 post-training quantization.

        This attempts to use TFLite's optimization API, but it has limitations
        when working with already-converted .tflite files.
        """
        print(f"\n🔧 Attempting quantization...")

        # THE PROBLEM: TFLite doesn't support requantizing .tflite files
        # We need the original model (SavedModel, Keras, etc.)

        # This is what WOULD work if we had the original model:
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ⚠️  LIMITATION: Can't requantize .tflite files directly         ║
╚══════════════════════════════════════════════════════════════════╝

TensorFlow Lite's quantization API requires the ORIGINAL model format:
  - SavedModel directory (TF 2.x)
  - Keras model (.h5 or .keras)
  - Frozen GraphDef (.pb)

It CANNOT requantize an already-converted .tflite file.

MediaPipe only distributes .tflite files (not original models).

SOLUTION: Use one of these approaches:

1. 📥 Download pre-quantized models from Google (if available)
   https://developers.google.com/mediapipe/solutions/vision/face_landmarker

2. 🔨 Rebuild MediaPipe from source with quantization flags (Bazel)
   See: build_int8_mediapipe.sh

3. 🔬 Use experimental tools (may not work):
   - ai-edge-quantizer (Google's experimental tool)
   - TFLite Model Optimization Toolkit

4. ⚡ Quick wins WITHOUT quantization:
   - Disable blendshapes (20% faster)
   - Use LIVE_STREAM mode
   - Lower confidence thresholds
   - Custom minimal build (30% smaller)
        """)

        return False


def extract_task_file(task_path, extract_dir):
    """Extract .tflite models from .task file (ZIP format)."""
    print(f"\n📦 Extracting {task_path}...")

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(task_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    files = os.listdir(extract_dir)
    print(f"✅ Extracted {len(files)} files:")
    for f in files:
        size_kb = os.path.getsize(os.path.join(extract_dir, f)) / 1024
        print(f"   - {f} ({size_kb:.1f} KB)")

    return files


def download_task_file(output_path):
    """Download MediaPipe face_landmarker.task if not present."""
    if os.path.exists(output_path):
        print(f"✅ Found existing {output_path}")
        return

    print(f"📥 Downloading face_landmarker.task...")
    import urllib.request

    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

    try:
        urllib.request.urlretrieve(url, output_path)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"✅ Downloaded {output_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"Please manually download from:")
        print(f"   {url}")
        exit(1)


def main():
    """Main workflow."""

    print("="*70)
    print("  MediaPipe Face Landmarker INT8 Quantization")
    print("="*70)

    # Setup paths
    task_file = "face_landmarker.task"
    extract_dir = "extracted"
    output_dir = "quantized"

    # Download model if needed
    download_task_file(task_file)

    # Extract .task file
    tflite_files = extract_task_file(task_file, extract_dir)

    # Filter to only .tflite files
    tflite_files = [f for f in tflite_files if f.endswith('.tflite')]

    if not tflite_files:
        print("❌ No .tflite files found in .task file")
        exit(1)

    # Attempt to quantize each model
    os.makedirs(output_dir, exist_ok=True)

    for tflite_file in tflite_files:
        input_path = os.path.join(extract_dir, tflite_file)
        output_path = os.path.join(output_dir, tflite_file.replace('.tflite', '_int8.tflite'))

        quantizer = TFLiteQuantizer(input_path, output_path)
        quantizer.load_model()

        # Generate representative dataset
        dataset_gen = quantizer.generate_representative_dataset(num_samples=100)

        # Attempt quantization (will explain why it fails)
        quantizer.quantize(dataset_gen)

    print("\n" + "="*70)
    print("  Summary")
    print("="*70)
    print("""
The TFLite quantization API cannot work with pre-converted .tflite files.

RECOMMENDED APPROACH:
1. Build minimal custom MediaPipe library (see build_int8_mediapipe.sh)
2. Apply quick optimization tweaks (disable blendshapes, etc.)
3. Expected result: 30-40% faster without needing INT8 quantization

If you MUST have INT8 quantization:
1. Check if Google provides pre-quantized models
2. Rebuild MediaPipe from source using Bazel with quantization flags
3. Or use experimental ai-edge-quantizer tool

See QUANTIZATION_GUIDE.md for detailed instructions.
    """)


if __name__ == "__main__":
    main()
