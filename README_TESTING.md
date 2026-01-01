# Testing AI Edge Quantizer on Different macOS

## Quick Start

Run this single command to test if AI Edge Quantizer works on your macOS:

```bash
./setup_and_test.sh
```

This will:
1. Create a clean Python 3.11 environment
2. Install ai-edge-quantizer-nightly
3. Apply compatibility patches
4. Test if it imports successfully
5. Show you exactly what works or what's broken

## Requirements

- **Python 3.11** (not 3.13 - ai-edge-quantizer doesn't support it yet)
- **macOS** (Linux also works but not tested here)
- **~500 MB disk space** (for TensorFlow + dependencies)

## Install Python 3.11 (if needed)

```bash
brew install python@3.11
```

## What to Expect

### If it works ✅

```
✅ SUCCESS! ai_edge_quantizer version: 0.5.0.dev20260101
✅ SUCCESS! quantizer and recipe imported
🎉 AI Edge Quantizer is WORKING!
```

### If it fails ❌

You'll see a detailed error message explaining:
- What specific component failed
- Whether it's the macOS version issue ("built for macOS 26.1")
- Whether it's a missing library
- Whether it's fixable or not

## Testing the Actual Quantization

If the import test succeeds, run the full quantization:

```bash
source venv_test/bin/activate
python quantize_with_ai_edge.py
```

This will:
1. Download MediaPipe face_landmarker.task (3.6 MB)
2. Extract the .tflite models
3. Quantize them to INT8
4. Repackage as face_landmarker_int8.task

## Known Issues

### macOS 15.4.1 (Sequoia)
- ❌ **ai-edge-litert-nightly** compiled for macOS 26.1 (doesn't exist)
- Result: `ImportError: built for macOS 26.1 which is newer than running OS`
- Status: **BROKEN** - Google build system bug

### macOS 15.5+ (Expected)
- 🟡 **Unknown** - May work if Google fixed the build
- Need testing on latest macOS

### macOS 14 (Sonoma)
- 🟡 **Unknown** - Older OS, might work with older builds

## Reporting Results

After running `./setup_and_test.sh`, please note:

1. **macOS version**: `sw_vers`
2. **Python version**: `python3.11 --version`
3. **Success or failure**: Copy the output
4. **Error message**: If it failed, copy the full error

## Files in This Directory

```
mediapipe_quantization/
├── README_TESTING.md              ← This file
├── setup_and_test.sh              ← Main test script ⭐
├── quantize_with_ai_edge.py       ← Full quantization script
├── .gitignore                     ← Excludes venv, models, etc.
├── QUANTIZATION_GUIDE.md          ← Technical background
├── QUANTIZATION_FAILURE_ANALYSIS.md ← Detailed failure analysis
└── (other documentation)
```

## What Gets Created

When you run the test:
```
venv_test/                    ← Python 3.11 virtual environment
  ├── bin/python              ← Python 3.11 executable
  ├── lib/python3.11/
  │   └── site-packages/
  │       ├── ai_edge_quantizer/
  │       ├── ai_edge_litert/
  │       │   └── tools/      ← Compatibility shim we create
  │       └── tensorflow/
  └── ...
```

## Cleanup

To remove everything:
```bash
rm -rf venv_test extracted quantized *.task
```

## Next Steps

If it **works** on your Mac:
- Great! You can quantize MediaPipe models
- Share your macOS version so we know it works

If it **fails** on your Mac:
- Check if it's the same "macOS 26.1" error
- We'll need to wait for Google to fix their build, or
- Build from source as last resort
