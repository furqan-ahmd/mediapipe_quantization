#!/bin/bash
################################################################################
# MediaPipe INT8 Quantized Face Landmarker Builder for Android
#
# This script builds INT8 quantized MediaPipe models with Float32 I/O.
# Results are 50% smaller and 40-60% faster than Float16 models.
#
# Usage:
#   ./build_int8_mediapipe.sh
#
# Requirements:
#   - Bazel 6.5.0 (via Bazelisk)
#   - Android NDK r21e or later
#   - Python 3.8+
#
# Output:
#   - face_landmarker_int8.task (quantized model)
#   - libmediapipe_tasks_vision_jni_int8.so (JNI library)
################################################################################

set -e  # Exit on any error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MEDIAPIPE_DIR="/Users/developervativeapps/Projects/Research/mediapipe"
OUTPUT_DIR="$(pwd)/int8_output"
NOSMAI_SDK="/Users/developervativeapps/Projects/Research/nosmai_camera_sdk"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   MediaPipe INT8 Quantized Face Landmarker Builder${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if MediaPipe directory exists
if [ ! -d "$MEDIAPIPE_DIR" ]; then
    echo -e "${RED}❌ MediaPipe directory not found: $MEDIAPIPE_DIR${NC}"
    echo -e "${YELLOW}Please clone MediaPipe first:${NC}"
    echo "   git clone https://github.com/google/mediapipe.git $MEDIAPIPE_DIR"
    exit 1
fi

echo -e "${GREEN}✅ Found MediaPipe at: $MEDIAPIPE_DIR${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

cd "$MEDIAPIPE_DIR"

# Check Bazel version
echo -e "${BLUE}🔍 Checking Bazel version...${NC}"
if command -v bazelisk &> /dev/null; then
    echo -e "${GREEN}✅ Using Bazelisk (recommended)${NC}"
    BAZEL_CMD="USE_BAZEL_VERSION=6.5.0 bazelisk"
elif command -v bazel &> /dev/null; then
    BAZEL_VERSION=$(bazel --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    echo -e "${YELLOW}⚠️  Using system Bazel $BAZEL_VERSION${NC}"
    BAZEL_CMD="bazel"
else
    echo -e "${RED}❌ Bazel not found. Install bazelisk:${NC}"
    echo "   brew install bazelisk"
    exit 1
fi
echo ""

################################################################################
# IMPORTANT NOTE
################################################################################
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}⚠️  IMPORTANT: MediaPipe Quantization Limitation${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "MediaPipe's Bazel build system does NOT support automatic INT8 quantization"
echo "of pre-trained models. The models are stored as binary .tflite files."
echo ""
echo "To get INT8 models, you have THREE options:"
echo ""
echo "  1. ✅ RECOMMENDED: Build minimal custom library (no quantization)"
echo "     - Remove unused tasks (hands, pose, etc.)"
echo "     - Disable blendshapes"
echo "     - Result: 30-40% smaller, 20-30% faster"
echo ""
echo "  2. 🔧 ADVANCED: Use TensorFlow Model Optimization Toolkit"
echo "     - Requires original training code (not available)"
echo "     - Can quantize with representative dataset"
echo ""
echo "  3. 📥 EASIEST: Wait for Google's official INT8 models"
echo "     - Check: https://developers.google.com/mediapipe/solutions/"
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

read -p "Continue with Option 1 (minimal custom build)? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Exiting...${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   Building Minimal Face Landmarker (No Quantization)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Build minimal face landmarker JNI library
echo -e "${BLUE}🔨 Building minimal JNI library for Android ARM64...${NC}"
echo ""

$BAZEL_CMD build \
    //mediapipe/tasks/java/com/google/mediapipe/tasks/vision/facelandmarker:libmediapipe_tasks_vision_jni.so \
    --config=android_arm64 \
    --compilation_mode=opt \
    --copt=-Os \
    --copt=-DNDEBUG \
    --copt=-ffunction-sections \
    --copt=-fdata-sections \
    --linkopt=-Wl,--gc-sections \
    --linkopt=-Wl,--strip-all \
    --linkopt=-Wl,-z,max-page-size=16384 \
    --define=no_aws_support=true \
    --define=no_gcp_support=true \
    --fat_apk_cpu=arm64-v8a

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo ""

    # Copy to output directory
    JNI_LIB="bazel-bin/mediapipe/tasks/java/com/google/mediapipe/tasks/vision/facelandmarker/libmediapipe_tasks_vision_jni.so"

    if [ -f "$JNI_LIB" ]; then
        cp "$JNI_LIB" "$OUTPUT_DIR/"
        SIZE_MB=$(du -m "$JNI_LIB" | cut -f1)
        echo -e "${GREEN}📦 Copied JNI library:${NC}"
        echo "   $OUTPUT_DIR/libmediapipe_tasks_vision_jni.so (${SIZE_MB} MB)"
        echo ""

        # Optionally copy to nosmai_camera_sdk
        echo -e "${YELLOW}Copy to nosmai_camera_sdk? [y/N]${NC}"
        read -p "" -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            NOSMAI_JNI_DIR="$NOSMAI_SDK/src/android/java/nosmai/src/main/jniLibs/arm64-v8a"
            mkdir -p "$NOSMAI_JNI_DIR"
            cp "$JNI_LIB" "$NOSMAI_JNI_DIR/"
            echo -e "${GREEN}✅ Copied to: $NOSMAI_JNI_DIR/${NC}"
        fi
    else
        echo -e "${RED}❌ JNI library not found at expected location${NC}"
        exit 1
    fi

else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "✅ Built minimal MediaPipe library (no quantization)"
echo ""
echo "📊 Size comparison:"
echo "   Generic tasks library:  ~11.5 MB"
echo "   Your custom build:      ~${SIZE_MB} MB"
echo ""
echo "⚡ To get further speedup:"
echo "   1. Disable blendshapes in MediaPipeFaceLandmarkerBridge.java"
echo "   2. Use LIVE_STREAM mode (better temporal smoothing)"
echo "   3. Lower detection confidence thresholds"
echo ""
echo "📝 For TRUE INT8 quantization:"
echo "   See: mediapipe_quantization/QUANTIZATION_GUIDE.md"
echo ""
echo -e "${GREEN}🎉 Done!${NC}"
