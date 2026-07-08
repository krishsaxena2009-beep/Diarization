# Testing & Debugging Guide for Robot Diarization

## Quick Start Workflow

```
Phone → Record Audio → Transfer to Laptop → Run Tests → Debug → Deploy to Robot
```

---

## **Phase 1: Recording Audio on Phone**

### Best Practices for Voice Samples

**Quality matters!** Record in these conditions:

✅ **DO:**
- Use quiet room (minimal background noise)
- Hold phone at consistent distance (6-12 inches from mouth)
- Record 30-60 seconds per speaker
- Have 2-4 different speakers for testing
- Speak naturally and clearly

❌ **DON'T:**
- Record in noisy environments (traffic, crowds)
- Whisper or mumble
- Record very short clips (<3 seconds)
- Use phone speaker (echo/feedback)

### Recommended Setup
```
Speaker 1: Hi, my name is John. This is a test of the speaker 
           diarization system. I hope this works well for your project.

Speaker 2: Hello everyone, this is Speaker 2 speaking. 
           The audio quality should be good for processing.

Speaker 1: Now I'm speaking again. This should be recognized as 
           the same speaker as before.
```

**File naming convention:**
```
test_audio_speaker1_speaker2.wav
or
meeting_001.m4a
or
conversation.wav
```

---

## **Phase 2: Transfer Audio to Laptop**

### Method 1: AirDrop (Mac + iPhone)
```bash
1. On Mac: Open Finder → AirDrop
2. On iPhone: Open Voice Memos → Select → Share → AirDrop
3. Select your Mac
4. Files appear in ~/Downloads/
```

### Method 2: Cloud Storage
```bash
# On iPhone: Upload to Google Drive/Dropbox
# On Mac: Download to a folder like ~/Downloads/audio_samples/

# Or via command line:
gsutil cp gs://your-bucket/audio.wav ~/Downloads/
aws s3 cp s3://your-bucket/audio.wav ~/Downloads/
```

### Method 3: Email/Messaging
```
Email the file to yourself, download on laptop
```

---

## **Phase 3: Setup Laptop Environment**

### Step 1: Clone Your Repository
```bash
cd ~/Desktop
git clone https://github.com/krishsaxena2009-beep/Diarization.git
cd Diarization
```

### Step 2: Create Data Directory
```bash
mkdir -p ~/datasets/audio_samples/
# Copy your phone recordings here
cp ~/Downloads/*.wav ~/datasets/audio_samples/
cp ~/Downloads/*.m4a ~/datasets/audio_samples/
```

### Step 3: Install Dependencies
```bash
bash setup.sh

# Or manually:
python3 -m venv diarization_env
source diarization_env/bin/activate
pip install -r requirements.txt
```

### Step 4: Update Configuration
Edit `hparams.yaml`:
```yaml
data_folder: /Users/yourname/datasets/audio_samples/
manual_annot_folder: /Users/yourname/datasets/annotations/
```

---

## **Phase 4: Test Your Setup**

### Run Diagnostic Test (Recommended First)
```bash
# This tests all components without processing full audio
python test_diarization.py ~/datasets/audio_samples/test_audio.wav
```

**Expected output:**
```
============================================================
SPEAKER DIARIZATION - DIAGNOSTIC TEST
============================================================

============================================================
STEP 1: Checking Audio File
============================================================
✓ Audio file loaded successfully
  - Sample rate: 16000 Hz
  - Channels: 1
  - Duration: 45.23 seconds
  - File size: 3.45 MB

============================================================
STEP 2: Checking Device
============================================================
⚠️  GPU not available. Using CPU (slower but will work).

============================================================
STEP 3: Loading Configuration
============================================================
✓ Configuration loaded from hparams.yaml
  - Sample rate: 16000 Hz
  - Mel bins: 80
  - Max subseg duration: 3.0s
  - Overlap: 1.5s
  - Affinity: cos

============================================================
STEP 4: Testing Embedding Extraction
============================================================
  Extracting Fbank features...
✓ Features extracted: shape torch.Size([1, 456, 80])
  - Time frames: 456
  - Feature dimension: 80
  Loading pre-trained ECAPA-TDNN model...
✓ Pre-trained weights loaded
✓ Embedding extracted: shape torch.Size([1, 192])
  - Embedding dimension: 192

============================================================
STEP 5: Testing Clustering
============================================================
  Creating test embeddings...
  Computing affinity matrix...
  Performing spectral clustering...
✓ Clustering successful
  - Labels: [0 0 0 0 0 1 1 1 1 1]
  - Unique speakers: 2

============================================================
TEST SUMMARY
============================================================
✓ All tests passed! Your setup is working.

You can now run:
  python diarize.py hparams.yaml test_audio.wav --output-rttm output.rttm
```

---

## **Phase 5: Run Full Diarization**

### Command
```bash
python diarize.py hparams.yaml ~/datasets/audio_samples/test_audio.wav \
  --output-rttm ~/datasets/output/test_audio.rttm
```

### Expected Output
```
============================================================
DIARIZATION RESULTS
============================================================
Audio file: /Users/yourname/datasets/audio_samples/test_audio.wav
Duration: 45.23s
Number of speakers: 2

RTTM Output:
SPEAKER audio 1 0.000 3.000 <NA> <NA> speaker_0 <NA>
SPEAKER audio 1 1.500 3.000 <NA> <NA> speaker_1 <NA>
SPEAKER audio 1 3.000 3.000 <NA> <NA> speaker_0 <NA>
SPEAKER audio 1 4.500 3.000 <NA> <NA> speaker_1 <NA>
...
============================================================
```

---

## **Common Issues & Fixes**

### Issue 1: "No module named 'speechbrain'"
```bash
# Fix:
pip install -r requirements.txt

# If that fails:
pip install --upgrade pip
pip install speechbrain torch torchaudio scikit-learn
```

### Issue 2: "Audio file not found"
```bash
# Check your path:
ls -la ~/datasets/audio_samples/

# Update hparams.yaml with correct path
```

### Issue 3: "CUDA out of memory" (GPU errors)
```bash
# The code automatically falls back to CPU, so this should work anyway
# But if slow, reduce batch_size in hparams.yaml:
batch_size: 256  # was 512
```

### Issue 4: "Wrong number of speakers detected"
```yaml
# Solution 1: Use oracle_n_spkrs (if you know the true count)
oracle_n_spkrs: True

# Solution 2: Set max_num_spkrs higher if more speakers
max_num_spkrs: 15  # was 10

# Solution 3: Adjust subsegment length
max_subseg_dur: 2.0  # was 3.0 (shorter = more segments)
```

### Issue 5: "Poor quality diarization output"
```yaml
# Try these adjustments:

# Better speaker separation:
affinity: 'nn'  # was 'cos' (nearest neighbor often better)

# Different clustering:
linkage: 'complete'  # was 'average' (more conservative)

# Finer subsegmentation:
max_subseg_dur: 1.5  # was 3.0
overlap: 0.75       # was 1.5
```

---

## **Phase 6: Evaluate Results**

### Check RTTM Output Format
```bash
# View the output:
cat ~/datasets/output/test_audio.rttm

# Count detected speakers:
cut -d' ' -f8 ~/datasets/output/test_audio.rttm | sort | uniq | wc -l
```

### Manual Listening Test
```bash
# Play audio and check if speaker labels match:
# Use VLC or similar to listen to audio
# Compare timeline with RTTM file
```

### Batch Processing Multiple Files
```bash
# Test with multiple audio files:
for file in ~/datasets/audio_samples/*.wav; do
    echo "Processing: $file"
    python diarize.py hparams.yaml "$file" \
      --output-rttm "~/datasets/output/$(basename $file .wav).rttm"
done
```

---

## **Phase 7: Prepare for Robot Integration**

Once tests pass, create a wrapper script for the robot:

```python
#!/usr/bin/env python3
# robot_diarizer.py

import sys
from diarize import SpeakerDiarizer
import speechbrain as sb

def diarize_audio(audio_path, output_rttm=None):
    """Simple wrapper for robot integration"""
    hparams = sb.load_hyperpyyaml('hparams.yaml')
    diarizer = SpeakerDiarizer(hparams)
    results = diarizer.diarize(audio_path, output_rttm)
    
    return {
        'speakers': results['num_speakers'],
        'duration': results['duration'],
        'rttm': results['rttm']
    }

if __name__ == "__main__":
    audio = sys.argv[1]
    result = diarize_audio(audio, output_rttm="output.rttm")
    print(f"Detected {result['speakers']} speakers")
```

---

## **Workflow Checklist**

- [ ] Record test audio on phone (2-4 speakers, 30-60s each)
- [ ] Transfer to laptop
- [ ] Update `hparams.yaml` paths
- [ ] Run `bash setup.sh` to install dependencies
- [ ] Run `python test_diarization.py audio.wav` to check setup
- [ ] Run `python diarize.py hparams.yaml audio.wav --output-rttm output.rttm`
- [ ] Check RTTM output looks reasonable
- [ ] Adjust parameters if needed
- [ ] Test with more audio samples
- [ ] Integrate into robot code

---

## **Next Steps**

Once this works:
1. **Collect more data** from different people
2. **Fine-tune parameters** for your use case
3. **Add predicted VAD** (for real-world robot use)
4. **Integrate into robot** ROS node or main code
5. **Test on robot hardware**

**Share any errors you encounter!** I can help debug. 🚀
