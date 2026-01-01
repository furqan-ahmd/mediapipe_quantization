#!/bin/bash
################################################################################
# AI Edge Quantizer Test Setup Script
#
# This creates a clean Python 3.11 environment and tests ai-edge-quantizer-nightly
#
# Usage:
#   ./setup_and_test.sh
################################################################################

set -e  # Exit on error

echo "================================"
echo "AI Edge Quantizer Test Setup"
echo "================================"
echo ""

# Clean up old venv
echo "🗑️  Removing old virtual environment..."
rm -rf venv_test
echo ""

# Create fresh venv with Python 3.11
echo "🐍 Creating Python 3.11 virtual environment..."
python3.11 -m venv venv_test
echo ""

# Activate and upgrade pip
echo "⬆️  Upgrading pip..."
./venv_test/bin/pip install --upgrade pip setuptools wheel --quiet
echo ""

# Install ai-edge-quantizer-nightly
echo "📦 Installing ai-edge-quantizer-nightly..."
echo "   (This will take a few minutes - downloading ~200MB TensorFlow)"
./venv_test/bin/pip install ai-edge-quantizer-nightly --quiet
echo ""

# Install tensorflow for compatibility
echo "📦 Installing tensorflow for compatibility..."
./venv_test/bin/pip install tensorflow --quiet
echo ""

# Create compatibility shim for missing tools module
echo "🔧 Creating compatibility shim for ai_edge_litert.tools..."
./venv_test/bin/python << 'PYTHON_EOF'
import ai_edge_litert
import os

ai_edge_litert_path = os.path.dirname(ai_edge_litert.__file__)
tools_dir = os.path.join(ai_edge_litert_path, 'tools')
os.makedirs(tools_dir, exist_ok=True)

# Create __init__.py that imports from tensorflow
with open(os.path.join(tools_dir, '__init__.py'), 'w') as f:
    f.write('# Compatibility shim for ai-edge-quantizer\n')
    f.write('from tensorflow.lite.tools import flatbuffer_utils\n')

print(f'✅ Created: {os.path.join(tools_dir, "__init__.py")}')
PYTHON_EOF
echo ""

# Test import
echo "================================"
echo "Testing ai-edge-quantizer import"
echo "================================"
echo ""

./venv_test/bin/python << 'PYTHON_EOF'
import sys
print("Python version:", sys.version)
print("")

try:
    print("Attempting to import ai_edge_quantizer...")
    import ai_edge_quantizer as aq
    print(f"✅ SUCCESS! ai_edge_quantizer version: {aq.__version__}")
    print("")

    print("Attempting to import submodules...")
    from ai_edge_quantizer import quantizer, recipe
    print("✅ SUCCESS! quantizer and recipe imported")
    print("")

    print("🎉 AI Edge Quantizer is WORKING!")
    print("")
    print("You can now use it like this:")
    print("  ./venv_test/bin/python")
    print("  >>> import ai_edge_quantizer as aq")
    print("  >>> from ai_edge_quantizer import quantizer, recipe")
    print("  >>> qt = quantizer.Quantizer('model.tflite')")
    print("  >>> qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())")
    print("  >>> qt.quantize().export_model('model_quantized.tflite')")

except ImportError as e:
    print(f"❌ FAILED: {e}")
    print("")
    print("This error indicates:")
    if "macOS 26.1" in str(e) or "newer than running OS" in str(e):
        print("  • The nightly build has corrupted macOS version metadata")
        print("  • Google's build system compiled .so files with wrong platform tag")
        print("  • This is NOT fixable without Google releasing a corrected build")
    elif "ai_edge_litert" in str(e):
        print("  • Missing or broken ai-edge-litert-nightly package")
        print("  • Try: pip install ai-edge-litert-nightly --upgrade")
    else:
        print("  • Unknown import error")
        print("  • Full error:", str(e))

    sys.exit(1)
PYTHON_EOF

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "To use the environment:"
echo "  source venv_test/bin/activate"
echo "  python"
echo ""
