#!/usr/bin/env python3
"""
Speaker Diarization Pipeline
Extracts speaker embeddings and performs clustering for speaker diarization.

Usage:
    python diarize.py hparams.yaml
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

import torch
import torchaudio
from sklearn.cluster import SpectralClustering

import speechbrain as sb
from speechbrain.utils.data_utils import load_pkl, save_pkl
from speechbrain.processing.features import Fbank, InputNormalization
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """
    Speaker Diarization class using ECAPA-TDNN embeddings and Spectral Clustering.
    """
    
    def __init__(self, hparams):
        """
        Initialize the diarizer with hyperparameters.
        
        Arguments
        ---------
        hparams : dict
            Dictionary containing hyperparameters from YAML config.
        """
        self.hparams = hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directories
        self._create_directories()
        
        # Initialize models and processors
        self._initialize_models()
        
        logger.info(f"Diarizer initialized on device: {self.device}")
    
    def _create_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.hparams.embedding_dir,
            self.hparams.meta_data_dir,
            self.hparams.sys_rttm_dir,
            self.hparams.der_dir,
        ]
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def _initialize_models(self):
        """Initialize neural network components."""
        # Feature computation (Fbank)
        self.compute_features = self.hparams.compute_features
        self.compute_features.to(self.device)
        
        # Normalization layers
        self.mean_var_norm = self.hparams.mean_var_norm
        self.mean_var_norm.to(self.device)
        
        self.mean_var_norm_emb = self.hparams.mean_var_norm_emb
        self.mean_var_norm_emb.to(self.device)
        
        # Embedding model (ECAPA-TDNN)
        self.embedding_model = self.hparams.embedding_model
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        
        # Load pre-trained weights
        self.hparams.pretrainer.collect_files()
        self.hparams.pretrainer.load_collected()
        logger.info("Pre-trained embedding model loaded successfully")
    
    def extract_embeddings(self, audio_path: str, duration: float = None) -> np.ndarray:
        """
        Extract speaker embeddings from audio file.
        
        Arguments
        ---------
        audio_path : str
            Path to audio file.
        duration : float, optional
            Duration in seconds. If None, uses entire file.
        
        Returns
        -------
        embeddings : np.ndarray
            Speaker embeddings of shape (n_subseg, emb_dim).
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.hparams.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.hparams.sampling_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Trim to duration if specified
        if duration is not None:
            num_samples = int(duration * self.hparams.sampling_rate)
            waveform = waveform[:, :num_samples]
        
        waveform = waveform.to(self.device)
        
        embeddings_list = []
        
        # Extract embeddings from subsegments
        with torch.no_grad():
            # Compute features
            features = self.compute_features(waveform)
            
            # Normalize features
            self.mean_var_norm.compute_current_stats(waveform)
            features = self.mean_var_norm(features, torch.ones(1).to(self.device))
            
            # Create subsegments
            subseg_dur = int(self.hparams.max_subseg_dur * self.hparams.sampling_rate)
            overlap = int(self.hparams.overlap * self.hparams.sampling_rate)
            hop = subseg_dur - overlap
            
            for start in range(0, waveform.shape[1] - subseg_dur, hop):
                subseg = waveform[:, start:start + subseg_dur]
                
                # Extract features for this subsegment
                feat_subseg = self.compute_features(subseg)
                
                # Extract embedding
                embedding = self.embedding_model(feat_subseg)
                
                # Normalize embedding
                self.mean_var_norm_emb.compute_current_stats(embedding)
                embedding = self.mean_var_norm_emb(embedding, torch.ones(1).to(self.device))
                
                embeddings_list.append(embedding.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings_list, axis=0)
        
        logger.info(f"Extracted {embeddings.shape[0]} embeddings from {audio_path}")
        return embeddings
    
    def cluster_speakers(self, embeddings: np.ndarray, n_speakers: int = None) -> np.ndarray:
        """
        Perform spectral clustering on embeddings.
        
        Arguments
        ---------
        embeddings : np.ndarray
            Speaker embeddings of shape (n_subseg, emb_dim).
        n_speakers : int, optional
            Number of speakers. If None, estimates automatically.
        
        Returns
        -------
        labels : np.ndarray
            Speaker labels for each subsegment.
        """
        # Normalize embeddings
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute affinity matrix based on cosine similarity
        if self.hparams.affinity == 'cos':
            affinity_matrix = np.dot(embeddings_norm, embeddings_norm.T)
        elif self.hparams.affinity == 'nn':
            # Nearest neighbor affinity
            distances = pdist(embeddings_norm, metric='cosine')
            affinity_matrix = squareform(1 - distances)
        else:
            raise ValueError(f"Unknown affinity: {self.hparams.affinity}")
        
        # Clip affinity to [0, 1]
        affinity_matrix = np.clip(affinity_matrix, 0, 1)
        
        # Estimate number of speakers if not provided
        if n_speakers is None:
            n_speakers = self._estimate_num_speakers(affinity_matrix)
            logger.info(f"Estimated number of speakers: {n_speakers}")
        
        # Perform spectral clustering
        clusterer = SpectralClustering(
            n_clusters=n_speakers,
            affinity='precomputed',
            linkage=self.hparams.linkage,
            eigen_solver=self.hparams.eigen_solver,
            random_state=self.hparams.seed
        )
        
        labels = clusterer.fit_predict(affinity_matrix)
        
        logger.info(f"Clustering complete: {n_speakers} speakers identified")
        return labels
    
    def _estimate_num_speakers(self, affinity_matrix: np.ndarray, max_speakers: int = None) -> int:
        """
        Estimate number of speakers using eigenvalue gap method.
        
        Arguments
        ---------
        affinity_matrix : np.ndarray
            Affinity matrix of shape (n_samples, n_samples).
        max_speakers : int, optional
            Maximum number of speakers to consider.
        
        Returns
        -------
        n_speakers : int
            Estimated number of speakers.
        """
        if max_speakers is None:
            max_speakers = self.hparams.max_num_spkrs
        
        # Convert affinity to Laplacian
        D = np.diag(affinity_matrix.sum(axis=1))
        L = D - affinity_matrix
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(L)
        eigenvalues = np.sort(eigenvalues)[:max_speakers]
        
        # Find the largest eigengap
        eigengaps = np.diff(eigenvalues)
        n_speakers = np.argmax(eigengaps) + 1
        
        return max(1, min(n_speakers, max_speakers))
    
    def embeddings_to_rttm(
        self,
        labels: np.ndarray,
        audio_duration: float,
        subseg_dur: float = None,
        overlap: float = None,
        output_path: str = None
    ) -> str:
        """
        Convert cluster labels to RTTM format.
        
        Arguments
        ---------
        labels : np.ndarray
            Speaker labels for each subsegment.
        audio_duration : float
            Total audio duration in seconds.
        subseg_dur : float, optional
            Subsegment duration. Uses hparams if None.
        overlap : float, optional
            Overlap between subsegments. Uses hparams if None.
        output_path : str, optional
            Path to save RTTM file.
        
        Returns
        -------
        rttm_content : str
            RTTM format string.
        """
        if subseg_dur is None:
            subseg_dur = self.hparams.max_subseg_dur
        if overlap is None:
            overlap = self.hparams.overlap
        
        rttm_lines = []
        hop = subseg_dur - overlap
        
        for i, label in enumerate(labels):
            start_time = i * hop
            end_time = min(start_time + subseg_dur, audio_duration)
            
            # RTTM format: SPEAKER <file> 1 <start> <duration> <conf> <model> <speaker> <null>
            duration = end_time - start_time
            rttm_line = f"SPEAKER audio 1 {start_time:.3f} {duration:.3f} <NA> <NA> speaker_{label} <NA>\n"
            rttm_lines.append(rttm_line)
        
        rttm_content = "".join(rttm_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(rttm_content)
            logger.info(f"RTTM file saved: {output_path}")
        
        return rttm_content
    
    def diarize(self, audio_path: str, output_rttm: str = None) -> Dict:
        """
        Complete diarization pipeline.
        
        Arguments
        ---------
        audio_path : str
            Path to audio file.
        output_rttm : str, optional
            Path to save output RTTM file.
        
        Returns
        -------
        results : dict
            Dictionary containing embeddings, labels, and RTTM content.
        """
        logger.info(f"Starting diarization for: {audio_path}")
        
        # Get audio duration
        waveform, sr = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sr
        
        # Extract embeddings
        embeddings = self.extract_embeddings(audio_path)
        
        # Determine number of speakers
        n_speakers = self.hparams.oracle_n_spkrs if self.hparams.oracle_n_spkrs else None
        
        # Cluster speakers
        labels = self.cluster_speakers(embeddings, n_speakers)
        
        # Convert to RTTM
        rttm_content = self.embeddings_to_rttm(
            labels,
            duration,
            output_path=output_rttm
        )
        
        results = {
            'audio_path': audio_path,
            'duration': duration,
            'embeddings': embeddings,
            'labels': labels,
            'rttm': rttm_content,
            'num_speakers': len(np.unique(labels))
        }
        
        logger.info(f"Diarization complete: {results['num_speakers']} speakers detected")
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Speaker Diarization Pipeline")
    parser.add_argument("hparams_file", type=str, help="Path to hyperparameters YAML file")
    parser.add_argument("audio_file", type=str, help="Path to audio file to diarize")
    parser.add_argument("--output-rttm", type=str, default=None, help="Output RTTM file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load hyperparameters
    hparams = sb.load_hyperpyyaml(args.hparams_file)
    
    # Initialize diarizer
    diarizer = SpeakerDiarizer(hparams)
    
    # Run diarization
    results = diarizer.diarize(args.audio_file, output_rttm=args.output_rttm)
    
    # Print results
    print("\n" + "="*60)
    print("DIARIZATION RESULTS")
    print("="*60)
    print(f"Audio file: {results['audio_path']}")
    print(f"Duration: {results['duration']:.2f}s")
    print(f"Number of speakers: {results['num_speakers']}")
    print(f"\nRTTM Output:\n{results['rttm']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
