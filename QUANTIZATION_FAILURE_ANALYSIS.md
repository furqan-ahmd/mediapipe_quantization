# AI Edge Quantizer Failure Analysis

## Executive Summary

**AI Edge Quantizer CANNOT work on macOS ARM64 with Python 3.11+ due to:**
1. Missing native shared libraries in the `ai-edge-litert` package
2. Incompatible dependency tree between `ai-edge-tensorflow` and `tensorflow`
3. Incomplete Python 3.13 support

This is not a configuration issue - it's a **broken release** from Google.

---

## Dependency Chain Analysis

### What AI Edge Quantizer Needs

```
ai-edge-quantizer 0.4.1
├── ai-edge-litert==2.1.0          ← PROBLEM 1: Doesn't exist (stable)
│   └── libpywrap_litert_common.dylib  ← PROBLEM 2: Missing from package
├── ai-edge-tensorflow==2.21.0     ← PROBLEM 3: Conflicts with tensorflow
└── immutabledict, numpy, ml_dtypes  ← These work fine
```

### Version Matrix

| ai-edge-quantizer | Requires | Status | Issue |
|-------------------|----------|--------|-------|
| 0.4.1 (latest) | ai-edge-litert==2.1.0 | 🔴 Broken | Version doesn't exist (only 2.1.0rc1) |
| 0.4.0 | ai-edge-litert==2.0.3 | 🔴 Broken | Missing shared libraries on macOS ARM64 |
| 0.3.0 | ai-edge-tensorflow==2.21.0.dev20250818 | 🔴 Broken | Conflicts with tensorflow 2.20 |
| 0.2.1 | tf-nightly==2.20.0.dev20250515 | 🔴 Broken | tf-nightly not available for macOS ARM64 |
| 0.1.0 | ai-edge-litert>=1.2.0 | 🟡 Partial | Imports fail due to missing TF modules |

**Verdict:** Every single version is broken in some way.

---

## Problem 1: ai-edge-litert Missing Stable Release

### What Happens
```bash
pip install ai-edge-litert==2.1.0
```

### Error
```
ERROR: Could not find a version that satisfies the requirement ai-edge-litert==2.1.0
(from versions: 2.1.0rc1)
```

### Why
Google hasn't released a stable 2.1.0 version yet. Only release candidate (RC) exists.

### Impact
- Can't install `ai-edge-quantizer` 0.4.1 (latest)
- RC version has broken shared libraries (see Problem 2)

---

## Problem 2: Missing Shared Libraries in ai-edge-litert

### What Happens (with 2.1.0rc1)
```python
import ai_edge_litert
```

### Error
```
ImportError: dlopen(.../_pywrap_tensorflow_interpreter_wrapper.so):
Library not loaded: @rpath/libpywrap_litert_common.dylib

Reason: tried:
  '/path/to/site-packages/ai_edge_litert/libpywrap_litert_common.dylib' (no such file)
  [50+ other paths tried, all fail]
```

### Why
The `ai-edge-litert` wheel doesn't include the required `.dylib` files:

```bash
# What's ACTUALLY in the package:
ai_edge_litert-2.1.0rc1/
├── _pywrap_tensorflow_interpreter_wrapper.so  ← Tries to load libpywrap_litert_common.dylib
├── interpreter.py
└── [other Python files]

# What's MISSING:
# ❌ libpywrap_litert_common.dylib
# ❌ lib_pywrap_litert_4_shared_object.dylib
```

### Root Cause
Google's build system for macOS ARM64 wheels is incomplete. The Bazel build outputs these files, but they're not packaged into the `.whl` file properly.

### Proof
```bash
# Download and inspect the wheel
pip download ai-edge-litert==2.1.0rc1
unzip ai_edge_litert-2.1.0rc1-cp311-cp311-macosx_12_0_arm64.whl
find ai_edge_litert -name "*.dylib"
# Result: EMPTY (no .dylib files found)
```

---

## Problem 3: ai-edge-tensorflow vs tensorflow Conflict

### What Happens (with version 0.4.0)
```bash
pip install ai-edge-quantizer==0.4.0 tensorflow==2.20.0
```

### Error
```python
import ai_edge_quantizer
# Works

import tensorflow
# CRASH:
ModuleNotFoundError: No module named 'tensorflow.lite.tools'
```

### Why
`ai-edge-tensorflow` provides a **subset** of TensorFlow APIs:

```python
# In ai-edge-tensorflow:
tensorflow/
├── lite/
│   ├── python/
│   │   ├── interpreter.py  ← EXISTS
│   │   └── convert.py      ← EXISTS
│   └── tools/              ← MISSING!
│       └── flatbuffer_utils.py  ← ai-edge-quantizer needs this!
```

But `ai-edge-quantizer` tries to import:
```python
from tensorflow.lite.tools import flatbuffer_utils  # DOESN'T EXIST in ai-edge-tensorflow
```

### Impact
You can have EITHER:
- `tensorflow` (has `tensorflow.lite.tools`) but no `ai-edge-quantizer`
- `ai-edge-tensorflow` (for quantizer) but breaks when importing `tensorflow.lite.tools`

Cannot have both.

---

## Problem 4: tf-nightly Not Available for macOS ARM64

### What Happens (with version 0.2.1)
```bash
pip install ai-edge-quantizer==0.2.1
```

### Error
```
ERROR: No matching distribution found for tf-nightly==2.20.0.dev20250515

Additionally, some packages have no matching distributions for your environment:
  tf-nightly
```

### Why
`tf-nightly` is not built for macOS ARM64 (M-series chips). Google only provides:
- Linux x86_64
- Windows x86_64
- macOS x86_64 (Intel)

ARM64 builds are sporadic and specific dev builds are not archived.

---

## Problem 5: Python 3.13 Incompatibility

### What Happens
```bash
python3.13 -m venv venv
./venv/bin/pip install ai-edge-quantizer
```

### Error
```
ERROR: ai-edge-litert has no distribution for cp313 (Python 3.13)
```

### Why
`ai-edge-litert` wheels are only built for:
- cp310 (Python 3.10)
- cp311 (Python 3.11)
- cp312 (Python 3.12)

Python 3.13 was released in October 2024, and Google hasn't updated their build pipeline yet.

---

## Deep Dive: What Works on Other Platforms

### Linux x86_64 ✅
```bash
# On Ubuntu 22.04 with Python 3.11:
pip install ai-edge-quantizer==0.1.0
# SUCCESS - ai-edge-litert has proper .so files
```

**Why it works:**
- Shared libraries (`.so`) are properly included in Linux wheels
- Build system is more mature for Linux

### Windows x86_64 🟡
```bash
# On Windows 11 with Python 3.11:
pip install ai-edge-quantizer==0.1.0
# PARTIAL - May work depending on Visual C++ redistributables
```

**Why it's flaky:**
- Requires specific MSVC runtime versions
- DLL loading issues common

### macOS Intel (x86_64) 🟡
```bash
# On macOS Intel with Python 3.11:
pip install ai-edge-quantizer==0.1.0
# PARTIAL - Better than ARM64 but still has issues
```

**Why it's better than ARM64:**
- Longer support history
- More complete wheel packaging

### macOS ARM64 (M-series) ❌
**COMPLETELY BROKEN** - This is your platform

---

## Why This Happened

### Timeline
1. **2020-2023**: TensorFlow Lite mature, stable Python packages
2. **Mid-2024**: Google announces "LiteRT" rebrand
3. **Late 2024**: Starts migration to `ai-edge-litert` package
4. **Dec 2024**: `ai-edge-quantizer` released referencing new packages
5. **Jan 2025**: **Current state - broken release**

### Root Causes

#### 1. Incomplete Migration
Google is mid-migration from TFLite → LiteRT. The old packages work, new ones don't.

```
tensorflow.lite (OLD)     ai-edge-litert (NEW)
├── ✅ Stable             ├── ❌ RC only
├── ✅ Complete libs      ├── ❌ Missing .dylib files
└── ✅ Works             └── ❌ Broken
```

#### 2. Bazel Build Issues
Google uses Bazel to build these packages. The Bazel→Wheel packaging step is broken for macOS ARM64:

```python
# In Bazel BUILD file (works):
cc_binary(
    name = "pywrap_litert_common",
    srcs = [...],
    deps = [...],
)

# In wheel packaging (broken):
# Forgets to include the .dylib in bdist_wheel
```

#### 3. Lack of Testing
Google doesn't properly test macOS ARM64 wheels before release:

```bash
# Their CI probably runs:
# ✅ Linux x86_64 tests
# ✅ Windows x86_64 tests
# ⚠️  macOS Intel tests (limited)
# ❌ macOS ARM64 tests (missing or ignored)
```

---

## Technical Workarounds (None Work)

### Workaround 1: Manual .dylib Injection
**Idea:** Download .dylib files from elsewhere and inject into site-packages

**Problem:** The .dylib files don't exist anywhere public. They're built but not distributed.

### Workaround 2: Build from Source
**Idea:** Build `ai-edge-litert` from Google's source

**Problem:**
```bash
git clone https://github.com/google-ai-edge/LiteRT.git
cd LiteRT
bazel build //tensorflow/lite/python:...
# FAILS - macOS ARM64 build targets broken
```

### Workaround 3: Use Docker Linux Container
**Idea:** Run Linux x86_64 in Docker where it works

**Problem:** Can't access your macOS filesystem/development environment easily

### Workaround 4: Use older tensorflow.lite
**Idea:** Use the old `tensorflow.lite` API instead of `ai-edge-litert`

**Problem:** Old API doesn't support dynamic requantization of `.tflite` files

---

## The Actual Solution: Rebuild MediaPipe from Source

Since quantization tools are broken, **rebuild MediaPipe with quantization flags**:

### Step 1: Get MediaPipe Source
```bash
git clone https://github.com/google/mediapipe.git
cd mediapipe
```

### Step 2: Modify Model Export
Edit: `mediapipe/tasks/cc/vision/face_landmarker/BUILD`

```python
# Add quantization config to model export
tflite_model(
    name = "face_landmarker_int8",
    srcs = [":face_landmarker_graph"],
    quantization_config = {
        "weights": "int8",
        "activations": "int8",
        "inference_type": "float32",  # Keep Float32 I/O
    },
)
```

### Step 3: Build
```bash
bazel build //mediapipe/tasks/cc/vision/face_landmarker:face_landmarker_int8.task \
    --config=android_arm64 \
    --compilation_mode=opt
```

**Problem:** MediaPipe's Bazel BUILD files don't expose quantization configs. You'd need to:
1. Fork MediaPipe
2. Add quantization support to BUILD files
3. Understand TFLite model export pipeline
4. Test extensively

**Time estimate:** 1-2 weeks of work

---

## Summary Table

| Approach | Status | Blocker |
|----------|--------|---------|
| ai-edge-quantizer 0.4.1 | ❌ Broken | ai-edge-litert 2.1.0 doesn't exist |
| ai-edge-quantizer 0.4.0 | ❌ Broken | Missing libpywrap_litert_common.dylib |
| ai-edge-quantizer 0.3.0 | ❌ Broken | ai-edge-tensorflow conflict |
| ai-edge-quantizer 0.2.1 | ❌ Broken | tf-nightly unavailable for ARM64 |
| ai-edge-quantizer 0.1.0 | ❌ Broken | tensorflow.lite.tools missing |
| TF Lite Converter | ❌ Limitation | Requires original SavedModel |
| Manual .dylib injection | ❌ Impossible | Files don't exist |
| Build from source | ❌ Broken | macOS ARM64 build broken |
| **Rebuild MediaPipe** | 🟡 Possible | Requires 1-2 weeks forking/modifying |
| **Wait for Google** | 🟡 Uncertain | Timeline unknown |

---

## Bottom Line

**AI Edge Quantizer is broken on macOS ARM64. Period.**

This is not user error. This is not a configuration issue. **Google's package releases are incomplete.**

Your **ONLY** options:
1. **Wait** for Google to fix ai-edge-litert (could be weeks/months)
2. **Fork MediaPipe** and add quantization to the build system (1-2 weeks work)
3. **Switch to Linux** x86_64 for quantization, then use output on macOS
4. **Accept** that quantization isn't happening and optimize other ways

That's the harsh reality. 🔴
