#!/usr/bin/env python3
"""
Crosstalk Analysis and Detection Module
Identifies, analyzes, and helps fix overlapping speaker problems.

Features:
- Detect regions with multiple speakers (crosstalk)
- Analyze embedding quality during overlap
- Visualize problematic segments
- Test different crosstalk handling strategies
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


@dataclass
class CrosstalkSegment:
    """Represents a crosstalk region"""
    start_time: float
    end_time: float
    duration: float
    num_speakers: int
    embedding_variance: float  # How mixed are the embeddings?
    confidence: float  # How confident are we there's crosstalk?
    affected_segments: List[int]  # Which subsegments are affected?


class CrosstalkAnalyzer:
    """
    Analyzes diarization output to identify and understand crosstalk problems.
    """
    
    def __init__(self, 
                 max_subseg_dur: float = 3.0,
                 overlap: float = 1.5):
        """
        Initialize analyzer.
        
        Arguments
        ---------
        max_subseg_dur : float
            Subsegment duration (should match hparams)
        overlap : float
            Overlap between subsegments
        """
        self.max_subseg_dur = max_subseg_dur
        self.overlap = overlap
        self.hop = max_subseg_dur - overlap
    
    def detect_crosstalk(self,
                        embeddings: np.ndarray,
                        labels: np.ndarray,
                        audio_duration: float,
                        threshold: float = 0.6) -> List[CrosstalkSegment]:
        """
        Detect potential crosstalk regions by analyzing embedding patterns.
        
        Arguments
        ---------
        embeddings : np.ndarray
            Speaker embeddings (n_subseg, emb_dim)
        labels : np.ndarray
            Cluster labels (n_subseg,)
        audio_duration : float
            Total audio duration in seconds
        threshold : float
            Confidence threshold for crosstalk detection (0-1)
        
        Returns
        -------
        crosstalk_regions : List[CrosstalkSegment]
            List of detected crosstalk segments
        """
        crosstalk_regions = []
        
        if embeddings.shape[0] < 2:
            return crosstalk_regions
        
        # Analyze embedding stability and label consistency
        embedding_stability = self._analyze_embedding_stability(embeddings)
        label_consistency = self._analyze_label_consistency(labels)
        
        # Combine signals to detect crosstalk
        combined_score = (embedding_stability + label_consistency) / 2
        
        # Find peaks in the crosstalk score
        peaks, properties = find_peaks(combined_score, height=threshold, distance=1)
        
        for peak_idx in peaks:
            start_time = peak_idx * self.hop
            end_time = min(start_time + self.max_subseg_dur, audio_duration)
            
            # Get affected subsegments
            affected = list(range(max(0, peak_idx - 1), min(len(labels), peak_idx + 2)))
            affected_labels = labels[affected]
            
            crosstalk_seg = CrosstalkSegment(
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                num_speakers=len(np.unique(affected_labels)),
                embedding_variance=embedding_stability[peak_idx],
                confidence=float(combined_score[peak_idx]),
                affected_segments=affected
            )
            
            crosstalk_regions.append(crosstalk_seg)
        
        logger.info(f"Detected {len(crosstalk_regions)} potential crosstalk regions")
        return crosstalk_regions
    
    def _analyze_embedding_stability(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Measure how 'pure' embeddings are (lower = more mixed/crosstalk).
        Uses variance of embeddings in sliding windows.
        
        Returns
        -------
        stability_score : np.ndarray
            Score for each subsegment (0-1, higher = more mixed)
        """
        n_segs = len(embeddings)
        stability = np.zeros(n_segs)
        
        # Normalize embeddings
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # For each subsegment, measure variance with neighbors
        for i in range(n_segs):
            neighbors = []
            for j in range(max(0, i - 1), min(n_segs, i + 2)):
                if i != j:
                    neighbors.append(embeddings_norm[j])
            
            if neighbors:
                # High cosine distance between neighbors = high mixing
                neighbor_embeddings = np.array(neighbors)
                
                # Compute pairwise distances
                distances = 1 - np.dot(embeddings_norm[i:i+1], neighbor_embeddings.T)[0]
                
                # Average distance indicates mixing
                stability[i] = np.mean(distances)
        
        return stability
    
    def _analyze_label_consistency(self, labels: np.ndarray) -> np.ndarray:
        """
        Measure label consistency (higher = more frequent speaker changes = possible crosstalk).
        
        Returns
        -------
        consistency_score : np.ndarray
            Score for each subsegment (0-1, higher = more unstable)
        """
        n_segs = len(labels)
        consistency = np.zeros(n_segs)
        
        for i in range(n_segs):
            # Check if label changes frequently around this point
            left = labels[max(0, i - 2):i]
            right = labels[i+1:min(n_segs, i + 3)]
            
            if len(left) > 0 and len(right) > 0:
                # Count unique labels in window
                window = np.concatenate([left, [labels[i]], right])
                unique_count = len(np.unique(window))
                
                # Normalize: max 3 speakers = score 1.0
                consistency[i] = min(1.0, unique_count / 3.0)
        
        return consistency
    
    def analyze_embedding_quality(self,
                                 embeddings: np.ndarray,
                                 labels: np.ndarray) -> Dict:
        """
        Analyze overall embedding quality and identify problematic subsegments.
        
        Returns
        -------
        quality_report : dict
            Metrics about embedding quality
        """
        report = {}
        
        # Normalize embeddings
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Intra-cluster homogeneity (embeddings from same speaker should be similar)
        intra_similarities = []
        for speaker_id in np.unique(labels):
            speaker_mask = labels == speaker_id
            speaker_embeddings = embeddings_norm[speaker_mask]
            
            if len(speaker_embeddings) > 1:
                # Compute pairwise similarities
                similarities = np.dot(speaker_embeddings, speaker_embeddings.T)
                # Get upper triangle (avoid diagonal)
                upper = similarities[np.triu_indices_from(similarities, k=1)]
                intra_similarities.extend(upper)
        
        # Inter-cluster separability (embeddings from different speakers should differ)
        inter_similarities = []
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    speaker_i = embeddings_norm[labels == unique_labels[i]]
                    speaker_j = embeddings_norm[labels == unique_labels[j]]
                    
                    if len(speaker_i) > 0 and len(speaker_j) > 0:
                        similarities = np.dot(speaker_i, speaker_j.T)
                        inter_similarities.extend(similarities.flatten())
        
        report['avg_intra_similarity'] = float(np.mean(intra_similarities)) if intra_similarities else 0.0
        report['avg_inter_similarity'] = float(np.mean(inter_similarities)) if inter_similarities else 0.0
        report['separability'] = float(report['avg_intra_similarity'] - report['avg_inter_similarity'])
        report['num_speakers'] = len(unique_labels)
        report['num_subsegments'] = len(embeddings)
        
        return report
    
    def find_problematic_segments(self,
                                 embeddings: np.ndarray,
                                 labels: np.ndarray,
                                 separability_threshold: float = 0.1) -> List[Tuple[int, float]]:
        """
        Find subsegments where embedding quality is poor (likely crosstalk).
        
        Returns
        -------
        problematic : List[Tuple[segment_idx, quality_score]]
            Subsegments with low quality scores
        """
        if embeddings.shape[0] < 2:
            return []
        
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        quality_scores = np.zeros(len(embeddings))
        
        for i in range(len(embeddings)):
            current_label = labels[i]
            
            # Compare with same-speaker embeddings
            same_speaker_mask = labels == current_label
            same_speaker_embeddings = embeddings_norm[same_speaker_mask]
            
            # Compare with different-speaker embeddings
            other_speaker_embeddings = embeddings_norm[~same_speaker_mask]
            
            if len(same_speaker_embeddings) > 1 and len(other_speaker_embeddings) > 0:
                # Similarity to own speaker
                intra_sim = np.mean(np.dot(embeddings_norm[i:i+1], same_speaker_embeddings.T))
                
                # Similarity to other speakers
                inter_sim = np.mean(np.dot(embeddings_norm[i:i+1], other_speaker_embeddings.T))
                
                # Quality = difference between intra and inter similarity
                quality = intra_sim - inter_sim
                quality_scores[i] = quality
        
        # Find segments below threshold
        problematic = [
            (i, float(quality_scores[i]))
            for i in range(len(quality_scores))
            if quality_scores[i] < separability_threshold
        ]
        
        return sorted(problematic, key=lambda x: x[1])
    
    def visualize_crosstalk(self,
                          embeddings: np.ndarray,
                          labels: np.ndarray,
                          crosstalk_regions: List[CrosstalkSegment],
                          audio_duration: float,
                          output_path: str = 'crosstalk_analysis.png'):
        """
        Create visualization of crosstalk regions.
        
        Arguments
        ---------
        embeddings : np.ndarray
            Speaker embeddings
        labels : np.ndarray
            Cluster labels
        crosstalk_regions : List[CrosstalkSegment]
            Detected crosstalk segments
        audio_duration : float
            Total audio duration
        output_path : str
            Path to save visualization
        """
        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 8))
            
            # Time axis
            times = np.arange(len(labels)) * self.hop
            
            # Plot 1: Speaker labels over time
            ax = axes[0]
            ax.scatter(times, labels, c=labels, cmap='tab10', s=50, alpha=0.7)
            ax.set_ylabel('Speaker ID')
            ax.set_title('Speaker Labels Over Time')
            ax.grid(True, alpha=0.3)
            
            # Highlight crosstalk regions
            for ct in crosstalk_regions:
                ax.axvspan(ct.start_time, ct.end_time, alpha=0.2, color='red', label='Crosstalk')
            
            # Plot 2: Embedding variance (stability)
            ax = axes[1]
            stability = self._analyze_embedding_stability(embeddings)
            ax.plot(times, stability, 'b-', linewidth=2)
            ax.fill_between(times, stability, alpha=0.3)
            ax.set_ylabel('Embedding Variance')
            ax.set_title('Embedding Stability (higher = more mixed)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            for ct in crosstalk_regions:
                ax.axvspan(ct.start_time, ct.end_time, alpha=0.2, color='red')
            
            # Plot 3: Label consistency
            ax = axes[2]
            consistency = self._analyze_label_consistency(labels)
            ax.plot(times, consistency, 'g-', linewidth=2)
            ax.fill_between(times, consistency, alpha=0.3)
            ax.set_ylabel('Label Inconsistency')
            ax.set_xlabel('Time (seconds)')
            ax.set_title('Label Consistency (higher = more speaker changes)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            for ct in crosstalk_regions:
                ax.axvspan(ct.start_time, ct.end_time, alpha=0.2, color='red')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Crosstalk visualization saved to {output_path}")
            plt.close()
        
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def generate_report(self,
                       embeddings: np.ndarray,
                       labels: np.ndarray,
                       crosstalk_regions: List[CrosstalkSegment],
                       audio_duration: float) -> str:
        """
        Generate a detailed text report about crosstalk issues.
        
        Returns
        -------
        report : str
            Formatted report
        """
        quality = self.analyze_embedding_quality(embeddings, labels)
        problematic = self.find_problematic_segments(embeddings, labels)
        
        report = []
        report.append("=" * 70)
        report.append("CROSSTALK ANALYSIS REPORT")
        report.append("=" * 70)
        
        report.append("\n[OVERALL QUALITY METRICS]")
        report.append(f"  Audio Duration: {audio_duration:.2f}s")
        report.append(f"  Number of Subsegments: {quality['num_subsegments']}")
        report.append(f"  Detected Speakers: {quality['num_speakers']}")
        report.append(f"  Intra-cluster Similarity: {quality['avg_intra_similarity']:.3f}")
        report.append(f"    (Higher = more consistent speaker embeddings)")
        report.append(f"  Inter-cluster Similarity: {quality['avg_inter_similarity']:.3f}")
        report.append(f"    (Lower = better speaker separation)")
        report.append(f"  Separability Score: {quality['separability']:.3f}")
        report.append(f"    (Higher = better separation, >0.1 is good)")
        
        report.append("\n[CROSSTALK DETECTION]")
        report.append(f"  Detected Regions: {len(crosstalk_regions)}")
        if crosstalk_regions:
            total_crosstalk = sum(ct.duration for ct in crosstalk_regions)
            pct = (total_crosstalk / audio_duration) * 100
            report.append(f"  Total Crosstalk Duration: {total_crosstalk:.2f}s ({pct:.1f}%)")
            
            report.append("\n  Crosstalk Segments:")
            for i, ct in enumerate(crosstalk_regions, 1):
                report.append(f"    {i}. {ct.start_time:.2f}s - {ct.end_time:.2f}s ({ct.duration:.2f}s)")
                report.append(f"       Confidence: {ct.confidence:.2f} | Speakers: {ct.num_speakers}")
        else:
            report.append("  No crosstalk regions detected!")
        
        report.append("\n[PROBLEMATIC SUBSEGMENTS]")
        if problematic:
            report.append(f"  Found {len(problematic)} subsegments with poor quality:")
            for seg_idx, quality_score in problematic[:10]:  # Show top 10
                time = seg_idx * self.hop
                report.append(f"    Segment {seg_idx} ({time:.2f}s): quality={quality_score:.3f}")
            if len(problematic) > 10:
                report.append(f"    ... and {len(problematic) - 10} more")
        else:
            report.append("  All subsegments have good quality!")
        
        report.append("\n[RECOMMENDATIONS FOR IMPROVEMENT]")
        
        if quality['separability'] < 0.05:
            report.append("  • LOW SEPARABILITY: Speakers are too similar")
            report.append("    - Try: Reduce max_subseg_dur (1.5s instead of 3.0s)")
            report.append("    - Try: Use different affinity metric (nn instead of cos)")
            report.append("    - Try: Increase number of expected speakers")
        
        if len(crosstalk_regions) > audio_duration * 0.2:  # >20% crosstalk
            report.append("  • HIGH CROSSTALK: Many overlapping regions")
            report.append("    - Try: Implement Voice Activity Detection (VAD)")
            report.append("    - Try: Use shorter subsegments (1.0s instead of 3.0s)")
            report.append("    - Try: Different clustering linkage (complete instead of average)")
        
        if len(problematic) > len(labels) * 0.3:  # >30% problematic
            report.append("  • POOR EMBEDDING QUALITY: Many segments don't separate well")
            report.append("    - Try: Pre-train model on more diverse data")
            report.append("    - Try: Improve audio quality (reduce noise)")
            report.append("    - Try: Implement speech enhancement before diarization")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("CROSSTALK ANALYZER - EXAMPLE USAGE")
    print("="*70)
    
    # Simulate embeddings and labels
    print("\nGenerating test data...")
    n_subsegments = 30
    embedding_dim = 192
    
    # Create embeddings for 2 speakers
    speaker_0_embeddings = np.random.randn(15, embedding_dim)
    speaker_1_embeddings = np.random.randn(15, embedding_dim)
    embeddings = np.vstack([speaker_0_embeddings, speaker_1_embeddings])
    
    # Create labels with some noise/crosstalk
    labels = np.array([0]*15 + [1]*15)
    # Add some label switching (simulates crosstalk confusion)
    labels[7:10] = 1  # False positives
    labels[22:25] = 0  # False positives
    
    audio_duration = n_subsegments * 1.5  # 45 seconds
    
    # Analyze
    analyzer = CrosstalkAnalyzer()
    crosstalk = analyzer.detect_crosstalk(embeddings, labels, audio_duration)
    
    print(f"\nDetected {len(crosstalk)} crosstalk regions")
    
    # Quality analysis
    quality = analyzer.analyze_embedding_quality(embeddings, labels)
    print(f"Separability: {quality['separability']:.3f}")
    
    # Generate report
    report = analyzer.generate_report(embeddings, labels, crosstalk, audio_duration)
    print("\n" + report)
    
    # Visualization
    analyzer.visualize_crosstalk(embeddings, labels, crosstalk, audio_duration)
