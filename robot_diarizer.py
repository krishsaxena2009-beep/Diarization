#!/usr/bin/env python3
"""
Robot Speaker Diarization Module
Handles live mic streams and recorded audio files.
Works with custom Python frameworks.

Features:
- Live microphone streaming
- Recorded file processing
- Real-time speaker tracking
- Callback-based event system
- Thread-safe operation
"""

import os
import queue
import threading
import logging
from pathlib import Path
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

import torch
import torchaudio
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering

import speechbrain as sb
from speechbrain.processing.features import InputNormalization

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """Represents a speaker segment in audio"""
    speaker_id: int
    start_time: float
    end_time: float
    duration: float
    confidence: float = 1.0
    
    def __str__(self):
        return f"Speaker {self.speaker_id}: {self.start_time:.2f}s - {self.end_time:.2f}s ({self.duration:.2f}s)"


@dataclass
class DiarizationResult:
    """Complete diarization output"""
    audio_path: str
    duration: float
    num_speakers: int
    segments: List[SpeakerSegment]
    rttm: str
    timestamp: datetime
    embeddings: np.ndarray
    labels: np.ndarray
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'audio_path': self.audio_path,
            'duration': self.duration,
            'num_speakers': self.num_speakers,
            'segments': [
                {
                    'speaker_id': s.speaker_id,
                    'start_time': s.start_time,
                    'end_time': s.end_time,
                    'duration': s.duration,
                    'confidence': s.confidence
                }
                for s in self.segments
            ],
            'rttm': self.rttm,
            'timestamp': self.timestamp.isoformat()
        }


class RobotDiarizer:
    """
    Main diarization class for robot integration.
    Handles both live audio and recorded files.
    """
    
    def __init__(self, hparams_path: str = 'hparams.yaml', device: str = None):
        """
        Initialize robot diarizer.
        
        Arguments
        ---------
        hparams_path : str
            Path to hyperparameters YAML file.
        device : str, optional
            Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.hparams = sb.load_hyperpyyaml(hparams_path)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"RobotDiarizer initialized on {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Event callbacks
        self.on_segment_detected = None  # Called when speaker segment detected
        self.on_complete = None  # Called when diarization complete
        self.on_error = None  # Called on error
    
    def _initialize_models(self):
        """Load and initialize all models."""
        # Feature extraction
        self.compute_features = self.hparams.compute_features.to(self.device)
        self.compute_features.eval()
        
        # Normalization
        self.mean_var_norm = self.hparams.mean_var_norm.to(self.device)
        self.mean_var_norm_emb = self.hparams.mean_var_norm_emb.to(self.device)
        
        # Embedding model
        self.embedding_model = self.hparams.embedding_model.to(self.device)
        self.embedding_model.eval()
        
        # Load pre-trained weights
        self.hparams.pretrainer.collect_files()
        self.hparams.pretrainer.load_collected()
        
        logger.info("All models loaded successfully")
    
    def diarize_file(self, audio_path: str, output_rttm: str = None) -> DiarizationResult:
        """
        Process a recorded audio file.
        
        Arguments
        ---------
        audio_path : str
            Path to audio file (WAV, MP3, M4A, etc.)
        output_rttm : str, optional
            Path to save RTTM output.
        
        Returns
        -------
        result : DiarizationResult
            Complete diarization results.
        """
        try:
            logger.info(f"Processing file: {audio_path}")
            
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sr
            
            # Extract embeddings
            embeddings = self._extract_embeddings(waveform, sr)
            
            # Cluster speakers
            labels = self._cluster_speakers(embeddings)
            
            # Generate segments
            segments = self._labels_to_segments(labels, duration)
            
            # Generate RTTM
            rttm = self._generate_rttm(segments)
            
            if output_rttm:
                with open(output_rttm, 'w') as f:
                    f.write(rttm)
                logger.info(f"RTTM saved to {output_rttm}")
            
            result = DiarizationResult(
                audio_path=audio_path,
                duration=duration,
                num_speakers=len(np.unique(labels)),
                segments=segments,
                rttm=rttm,
                timestamp=datetime.now(),
                embeddings=embeddings,
                labels=labels
            )
            
            if self.on_complete:
                self.on_complete(result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            if self.on_error:
                self.on_error(str(e))
            raise
    
    def diarize_stream(self, 
                      audio_stream: queue.Queue,
                      chunk_duration: float = 3.0,
                      overlap_duration: float = 1.5) -> None:
        """
        Process live audio stream from microphone.
        
        Arguments
        ---------
        audio_stream : queue.Queue
            Queue containing audio chunks. Each item should be:
            (waveform_tensor, sample_rate) or None to stop.
        chunk_duration : float
            Duration of each processing chunk (seconds).
        overlap_duration : float
            Overlap between chunks (seconds).
        
        Example
        -------
        >>> diarizer = RobotDiarizer()
        >>> audio_queue = queue.Queue()
        >>> 
        >>> # In microphone thread:
        >>> def mic_stream():
        ...     while True:
        ...         chunk = get_audio_from_mic()
        ...         audio_queue.put((chunk, 16000))
        >>> 
        >>> # Process in main thread:
        >>> diarizer.diarize_stream(audio_queue)
        """
        try:
            logger.info("Starting live stream diarization")
            
            audio_buffer = []
            sample_rate = None
            total_samples = 0
            
            while True:
                try:
                    # Get audio chunk from queue (timeout to prevent hanging)
                    item = audio_stream.get(timeout=1.0)
                    
                    # None signals stream end
                    if item is None:
                        logger.info("Stream ended")
                        break
                    
                    waveform, sr = item
                    sample_rate = sr
                    
                    # Accumulate audio
                    audio_buffer.append(waveform)
                    total_samples += waveform.shape[1]
                    
                    # Process when we have enough audio
                    chunk_samples = int(chunk_duration * sample_rate)
                    
                    if total_samples >= chunk_samples:
                        # Concatenate buffer
                        full_audio = torch.cat(audio_buffer, dim=1)
                        
                        # Extract embeddings for this chunk
                        embeddings = self._extract_embeddings(full_audio, sample_rate)
                        
                        # Cluster
                        labels = self._cluster_speakers(embeddings)
                        
                        # Generate segments
                        current_duration = full_audio.shape[1] / sample_rate
                        segments = self._labels_to_segments(labels, current_duration)
                        
                        # Call callback for each segment
                        if self.on_segment_detected:
                            for segment in segments:
                                self.on_segment_detected(segment)
                        
                        logger.info(f"Processed {current_duration:.2f}s, detected {len(np.unique(labels))} speakers")
                        
                        # Keep overlap for next iteration
                        overlap_samples = int(overlap_duration * sample_rate)
                        if full_audio.shape[1] > overlap_samples:
                            audio_buffer = [full_audio[:, -overlap_samples:]]
                            total_samples = overlap_samples
                        else:
                            audio_buffer = [full_audio]
                
                except queue.Empty:
                    # No new data, but keep waiting
                    continue
        
        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
            if self.on_error:
                self.on_error(str(e))
            raise
    
    def _extract_embeddings(self, waveform: torch.Tensor, sr: int) -> np.ndarray:
        """Extract speaker embeddings from waveform."""
        # Resample if needed
        if sr != self.hparams.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.hparams.sampling_rate)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        waveform = waveform.to(self.device)
        embeddings_list = []
        
        with torch.no_grad():
            # Process subsegments
            subseg_dur = int(self.hparams.max_subseg_dur * self.hparams.sampling_rate)
            overlap = int(self.hparams.overlap * self.hparams.sampling_rate)
            hop = subseg_dur - overlap
            
            for start in range(0, max(1, waveform.shape[1] - subseg_dur + 1), hop):
                end = min(start + subseg_dur, waveform.shape[1])
                subseg = waveform[:, start:end]
                
                # Extract features
                features = self.compute_features(subseg)
                
                # Extract embedding
                embedding = self.embedding_model(features)
                embeddings_list.append(embedding.cpu().numpy())
        
        return np.concatenate(embeddings_list, axis=0) if embeddings_list else np.array([]).reshape(0, 192)
    
    def _cluster_speakers(self, embeddings: np.ndarray, n_speakers: int = None) -> np.ndarray:
        """Perform spectral clustering on embeddings."""
        if embeddings.shape[0] == 0:
            return np.array([])
        
        # Normalize embeddings
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute affinity matrix
        if self.hparams.affinity == 'cos':
            affinity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        else:  # 'nn'
            distances = pdist(embeddings_norm, metric='cosine')
            affinity_matrix = squareform(1 - distances)
        
        affinity_matrix = np.clip(affinity_matrix, 0, 1)
        
        # Estimate speakers if not provided
        if n_speakers is None:
            n_speakers = self._estimate_num_speakers(affinity_matrix)
        
        # Cluster
        clusterer = SpectralClustering(
            n_clusters=max(1, n_speakers),
            affinity='precomputed',
            linkage=self.hparams.linkage,
            eigen_solver=self.hparams.eigen_solver,
            random_state=self.hparams.seed
        )
        
        return clusterer.fit_predict(affinity_matrix)
    
    def _estimate_num_speakers(self, affinity_matrix: np.ndarray) -> int:
        """Estimate number of speakers using eigenvalue gap."""
        max_speakers = self.hparams.max_num_spkrs
        
        D = np.diag(affinity_matrix.sum(axis=1))
        L = D - affinity_matrix
        
        eigenvalues = np.linalg.eigvalsh(L)
        eigenvalues = np.sort(eigenvalues)[:max_speakers]
        
        eigengaps = np.diff(eigenvalues)
        n_speakers = np.argmax(eigengaps) + 1 if len(eigengaps) > 0 else 1
        
        return max(1, min(n_speakers, max_speakers))
    
    def _labels_to_segments(self, labels: np.ndarray, duration: float) -> List[SpeakerSegment]:
        """Convert cluster labels to speaker segments."""
        segments = []
        
        subseg_dur = self.hparams.max_subseg_dur
        overlap = self.hparams.overlap
        hop = subseg_dur - overlap
        
        for i, label in enumerate(labels):
            start_time = i * hop
            end_time = min(start_time + subseg_dur, duration)
            
            segments.append(SpeakerSegment(
                speaker_id=int(label),
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time
            ))
        
        return segments
    
    def _generate_rttm(self, segments: List[SpeakerSegment]) -> str:
        """Generate RTTM format output."""
        rttm_lines = []
        for segment in segments:
            line = f"SPEAKER audio 1 {segment.start_time:.3f} {segment.duration:.3f} <NA> <NA> speaker_{segment.speaker_id} <NA>\n"
            rttm_lines.append(line)
        return "".join(rttm_lines)


class MicrophoneStream:
    """
    Simple microphone stream capture for live audio.
    Can be replaced with your robot's actual microphone interface.
    """
    
    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 0.1):
        """
        Initialize microphone stream.
        
        Arguments
        ---------
        sample_rate : int
            Sample rate in Hz.
        chunk_duration : float
            Duration of each audio chunk in seconds.
        """
        try:
            import pyaudio
            self.pyaudio = pyaudio.PyAudio()
        except ImportError:
            logger.warning("pyaudio not installed. Install with: pip install pyaudio")
            self.pyaudio = None
        
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.stream = None
    
    def start(self) -> queue.Queue:
        """
        Start microphone stream.
        
        Returns
        -------
        audio_queue : queue.Queue
            Queue containing (waveform, sample_rate) tuples.
        """
        if not self.pyaudio:
            raise RuntimeError("pyaudio is required for microphone input. Install with: pip install pyaudio")
        
        audio_queue = queue.Queue()
        
        self.stream = self.pyaudio.open(
            format=self.pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_samples
        )
        
        # Start background thread to capture audio
        def capture():
            try:
                while self.stream.is_active():
                    data = self.stream.read(self.chunk_samples, exception_on_overflow=False)
                    audio_np = np.frombuffer(data, dtype=np.float32)
                    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                    audio_queue.put((audio_tensor, self.sample_rate))
            except Exception as e:
                logger.error(f"Microphone error: {e}")
                audio_queue.put(None)
        
        thread = threading.Thread(target=capture, daemon=True)
        thread.start()
        
        logger.info("Microphone stream started")
        return audio_queue
    
    def stop(self):
        """Stop microphone stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            logger.info("Microphone stream stopped")


# ============================================
# Example Usage for Robot Integration
# ============================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Process recorded file
    print("\n" + "="*60)
    print("EXAMPLE 1: Process Recorded File")
    print("="*60)
    
    diarizer = RobotDiarizer('hparams.yaml')
    
    # Set callbacks
    diarizer.on_complete = lambda result: print(f"\n✓ Complete: {result.num_speakers} speakers detected")
    diarizer.on_error = lambda e: print(f"✗ Error: {e}")
    
    # Process file
    result = diarizer.diarize_file('test_audio.wav', output_rttm='output.rttm')
    print(f"\nResults:")
    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Speakers: {result.num_speakers}")
    print(f"  Segments:")
    for segment in result.segments[:5]:  # Show first 5
        print(f"    {segment}")
    
    # Example 2: Live microphone stream (uncomment to test)
    # print("\n" + "="*60)
    # print("EXAMPLE 2: Live Microphone Stream")
    # print("="*60)
    # print("Listening to microphone... (press Ctrl+C to stop)")
    # 
    # mic = MicrophoneStream(sample_rate=16000)
    # audio_queue = mic.start()
    # 
    # diarizer.on_segment_detected = lambda seg: print(f"  {seg}")
    # 
    # try:
    #     diarizer.diarize_stream(audio_queue, chunk_duration=5.0)
    # except KeyboardInterrupt:
    #     audio_queue.put(None)
    #     mic.stop()
