# Speaker Diarization Baseline

## Overview
This repository implements a speaker diarization system using:
- **Embeddings**: ECAPA-TDNN (pre-trained on VoxCeleb)
- **Clustering**: Spectral Clustering with cosine affinity
- **Dataset**: AMI Corpus (optional)

## Quick Start

### 1. Install Dependencies
```bash
# Option A: Using virtual environment (recommended)
bash setup.sh

# Option B: Manual installation
pip install -r requirements.txt
```

### 2. Update Configuration
Edit `hparams.yaml` and replace placeholder paths:
```yaml
data_folder: /path/to/amicorpus/
manual_annot_folder: /path/to/ami_public_manual_1.6.2/
```

### 3. Run Diarization
```bash
python diarize.py hparams.yaml your_audio.wav --output-rttm output.rttm
```

## Configuration

### Key Parameters in `hparams.yaml`

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_subseg_dur` | Max duration of subsegment (seconds) | 3.0 |
| `overlap` | Overlap between subsegments (seconds) | 1.5 |
| `affinity` | Affinity metric for clustering | `cos` |
| `oracle_n_spkrs` | Use oracle speaker count | `True` |
| `max_num_spkrs` | Maximum speakers to detect | 10 |

## Output

The diarization script outputs:
- **RTTM file**: Speaker segmentation in standard RTTM format
- **Embeddings**: Speaker embeddings saved to `results/ami/ecapa/save/embeddings/`
- **Logs**: Pipeline execution logs

### RTTM Format Example
```
SPEAKER audio 1 0.000 2.500 <NA> <NA> speaker_0 <NA>
SPEAKER audio 1 2.500 3.000 <NA> <NA> speaker_1 <NA>
SPEAKER audio 1 5.000 1.500 <NA> <NA> speaker_0 <NA>
```

## File Structure
```
.
├── diarize.py              # Main diarization pipeline
├── hparams.yaml            # Hyperparameter configuration
├── requirements.txt        # Python dependencies
├── setup.sh               # Automated setup script
└── README.md              # This file
```

## Dependencies
- **speechbrain**: SpeechBrain framework
- **torch / torchaudio**: Deep learning framework
- **scikit-learn**: Machine learning library for clustering
- **scipy / numpy**: Scientific computing

See `requirements.txt` for exact versions.

## Future Improvements
- [ ] Predicted VAD (Voice Activity Detection) for real-world use
- [ ] Batch processing for multiple files
- [ ] DER (Diarization Error Rate) evaluation metrics
- [ ] Clustering threshold optimization
- [ ] Multi-mic beamforming support

## References
- ECAPA-TDNN: [arxiv.org/abs/2005.07143](https://arxiv.org/abs/2005.07143)
- SpeechBrain: [speechbrain.github.io](https://speechbrain.github.io/)
- AMI Corpus: [groups.inf.ed.ac.uk/ami](http://groups.inf.ed.ac.uk/ami/)

## Authors
- Original baseline: Nauman Dawalatabad (2020)
- Improvements & Python implementation: [Your contributions]

## License
[Specify your license here]
