#!/usr/bin/env python3
"""
Quick Test Script for Speaker Diarization
Tests audio input, processes it, and provides detailed feedback.

Usage:
    python test_diarization.py audio.wav
"""

import os
import sys
import logging
from pathlib import Path

try:
    import torch
    import torchaudio
    import numpy as np
    from sklearn.cluster import SpectralClustering
    import speechbrain as sb
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_audio_file(audio_path):
    """Verify audio file exists and is readable."""
    print("\n" + "="*60)
    print("STEP 1: Checking Audio File")
    print("="*60)
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return False
    
    try:
        waveform, sr = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sr
        channels = waveform.shape[0]
        
        print(f"✓ Audio file loaded successfully")
        print(f"  - Sample rate: {sr} Hz")
        print(f"  - Channels: {channels}")
        print(f"  - Duration: {duration:.2f} seconds")
        print(f"  - File size: {os.path.getsize(audio_path) / 1e6:.2f} MB")
        
        if duration < 3:
            print(f"⚠️  Warning: Audio is very short ({duration:.2f}s). Minimum recommended is 3s.")
        
        return True
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return False

def check_device():
    """Check if GPU is available."""
    print("\n" + "="*60)
    print("STEP 2: Checking Device")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
    else:
        print("⚠️  GPU not available. Using CPU (slower but will work).")
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_hparams(hparams_path="hparams.yaml"):
    """Load hyperparameters from YAML."""
    print("\n" + "="*60)
    print("STEP 3: Loading Configuration")
    print("="*60)
    
    if not os.path.exists(hparams_path):
        print(f"❌ Config file not found: {hparams_path}")
        return None
    
    try:
        hparams = sb.load_hyperpyyaml(hparams_path)
        print(f"✓ Configuration loaded from {hparams_path}")
        print(f"  - Sample rate: {hparams.sampling_rate} Hz")
        print(f"  - Mel bins: {hparams.n_mels}")
        print(f"  - Max subseg duration: {hparams.max_subseg_dur}s")
        print(f"  - Overlap: {hparams.overlap}s")
        print(f"  - Affinity: {hparams.affinity}")
        return hparams
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def test_embedding_extraction(audio_path, hparams, device):
    """Test embedding extraction."""
    print("\n" + "="*60)
    print("STEP 4: Testing Embedding Extraction")
    print("="*60)
    
    try:
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != hparams.sampling_rate:
            print(f"  Resampling from {sr} to {hparams.sampling_rate} Hz...")
            resampler = torchaudio.transforms.Resample(sr, hparams.sampling_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            print(f"  Converting {waveform.shape[0]} channels to mono...")
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.to(device)
        
        # Extract features
        print(f"  Extracting Fbank features...")
        compute_features = hparams.compute_features.to(device)
        compute_features.eval()
        
        with torch.no_grad():
            features = compute_features(waveform)
        
        print(f"✓ Features extracted: shape {features.shape}")
        print(f"  - Time frames: {features.shape[1]}")
        print(f"  - Feature dimension: {features.shape[2]}")
        
        # Test embedding model
        print(f"  Loading pre-trained ECAPA-TDNN model...")
        embedding_model = hparams.embedding_model.to(device)
        embedding_model.eval()
        
        # Load weights
        hparams.pretrainer.collect_files()
        hparams.pretrainer.load_collected()
        print(f"✓ Pre-trained weights loaded")
        
        # Extract embedding (use first 3 seconds)
        subseg_dur = int(hparams.max_subseg_dur * hparams.sampling_rate)
        subseg = waveform[:, :min(subseg_dur, waveform.shape[1])]
        
        with torch.no_grad():
            feat_subseg = compute_features(subseg)
            embedding = embedding_model(feat_subseg)
        
        print(f"✓ Embedding extracted: shape {embedding.shape}")
        print(f"  - Embedding dimension: {embedding.shape[1]}")
        
        return True
    except Exception as e:
        print(f"❌ Error during embedding extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clustering():
    """Test clustering module."""
    print("\n" + "="*60)
    print("STEP 5: Testing Clustering")
    print("="*60)
    
    try:
        # Create dummy embeddings for 2 speakers
        print("  Creating test embeddings...")
        embeddings_speaker1 = np.random.randn(5, 192)  # 5 subsegments, speaker 1
        embeddings_speaker2 = np.random.randn(5, 192)  # 5 subsegments, speaker 2
        embeddings = np.vstack([embeddings_speaker1, embeddings_speaker2])
        
        # Normalize
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute affinity
        print("  Computing affinity matrix...")
        affinity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        affinity_matrix = np.clip(affinity_matrix, 0, 1)
        
        # Cluster
        print("  Performing spectral clustering...")
        clusterer = SpectralClustering(
            n_clusters=2,
            affinity='precomputed',
            linkage='average',
            eigen_solver='arpack'
        )
        labels = clusterer.fit_predict(affinity_matrix)
        
        print(f"✓ Clustering successful")
        print(f"  - Labels: {labels}")
        print(f"  - Unique speakers: {len(np.unique(labels))}")
        
        return True
    except Exception as e:
        print(f"❌ Error during clustering: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_diarization.py <audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    
    print("\n" + "="*60)
    print("SPEAKER DIARIZATION - DIAGNOSTIC TEST")
    print("="*60)
    
    # Run tests
    tests = [
        ("Audio File", lambda: check_audio_file(audio_path)),
        ("Device", lambda: check_device() is not None),
        ("Configuration", lambda: load_hparams() is not None),
    ]
    
    results = []
    hparams = load_hparams()
    device = check_device()
    
    if check_audio_file(audio_path) and hparams and device:
        results.append(("Embedding Extraction", test_embedding_extraction(audio_path, hparams, device)))
        results.append(("Clustering", test_clustering()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("✓ All tests passed! Your setup is working.")
        print("\nYou can now run:")
        print(f"  python diarize.py hparams.yaml {audio_path} --output-rttm output.rttm")
    else:
        print("❌ Some tests failed. Check errors above.")
        print("\nCommon fixes:")
        print("  1. Update hparams.yaml with correct paths")
        print("  2. Make sure audio file format is supported (WAV, MP4, etc.)")
        print("  3. Run: pip install -r requirements.txt")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
