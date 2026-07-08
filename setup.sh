#!/bin/bash
# Setup script for Speaker Diarization pipeline

echo "=================================================="
echo "Speaker Diarization Pipeline - Setup"
echo "=================================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment (optional but recommended)
echo ""
echo "Creating virtual environment..."
python3 -m venv diarization_env
source diarization_env/bin/activate

echo "✓ Virtual environment created and activated"

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Update hparams.yaml with your actual data paths:"
echo "   - data_folder: /path/to/amicorpus/"
echo "   - manual_annot_folder: /path/to/ami_public_manual_1.6.2/"
echo ""
echo "2. Run diarization:"
echo "   python diarize.py hparams.yaml your_audio.wav --output-rttm output.rttm"
echo ""
echo "To deactivate the virtual environment later, run:"
echo "   deactivate"
echo ""
