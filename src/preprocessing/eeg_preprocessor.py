"""
EEG Preprocessor for Parkinson's Disease Detection

This module provides comprehensive preprocessing functionality for EEG data,
including filtering, artifact removal, segmentation, and normalization.
"""

import numpy as np
import pandas as pd
import mne
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import yaml
import os
from typing import Tuple, List, Optional, Dict, Any
import logging

from .signal_filtering import SignalFilter
from .artifact_removal import ArtifactRemover

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """
    Comprehensive EEG preprocessing pipeline for Parkinson's disease detection.
    
    This class handles the complete preprocessing workflow including:
    - Signal filtering (notch, bandpass)
    - Artifact removal
    - Signal segmentation
    - Feature normalization
    - Data augmentation
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the EEG preprocessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.signal_filter = SignalFilter(self.config['eeg'])
        self.artifact_remover = ArtifactRemover()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default settings.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration settings."""
        return {
            'eeg': {
                'sampling_rate': 250,
                'channels': ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", 
                           "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"],
                'notch_filter': 50,
                'bandpass_filter': [0.5, 45],
                'segment_length': 10,
                'overlap': 0.5
            }
        }
    
    def preprocess_raw_data(self, eeg_data: np.ndarray, 
                          channels: Optional[List[str]] = None,
                          sampling_rate: Optional[int] = None) -> np.ndarray:
        """
        Preprocess raw EEG data through the complete pipeline.
        
        Args:
            eeg_data: Raw EEG data (channels x samples)
            channels: Channel names (optional)
            sampling_rate: Sampling rate in Hz (optional)
            
        Returns:
            Preprocessed EEG data
        """
        logger.info("Starting EEG preprocessing pipeline...")
        
        # Use config values if not provided
        if channels is None:
            channels = self.config['eeg']['channels']
        if sampling_rate is None:
            sampling_rate = self.config['eeg']['sampling_rate']
            
        # Step 1: Basic signal filtering
        logger.info("Applying signal filters...")
        filtered_data = self.signal_filter.apply_filters(eeg_data, sampling_rate)
        
        # Step 2: Artifact removal
        logger.info("Removing artifacts...")
        clean_data = self.artifact_remover.remove_artifacts(filtered_data, sampling_rate)
        
        # Step 3: Segment the data
        logger.info("Segmenting data...")
        segments = self._segment_data(clean_data, sampling_rate)
        
        # Step 4: Normalize segments
        logger.info("Normalizing segments...")
        normalized_segments = self._normalize_segments(segments)
        
        logger.info(f"Preprocessing complete. Generated {len(normalized_segments)} segments.")
        return normalized_segments
    
    def _segment_data(self, eeg_data: np.ndarray, sampling_rate: int) -> List[np.ndarray]:
        """
        Segment EEG data into overlapping windows.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            List of segmented data arrays
        """
        segment_length = self.config['eeg']['segment_length']
        overlap = self.config['eeg']['overlap']
        
        # Calculate segment parameters
        segment_samples = int(segment_length * sampling_rate)
        step_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        n_samples = eeg_data.shape[1]
        
        for start in range(0, n_samples - segment_samples + 1, step_samples):
            end = start + segment_samples
            segment = eeg_data[:, start:end]
            segments.append(segment)
            
        return segments
    
    def _normalize_segments(self, segments: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalize EEG segments using z-score normalization.
        
        Args:
            segments: List of EEG segments
            
        Returns:
            List of normalized segments
        """
        normalized_segments = []
        
        for segment in segments:
            # Apply z-score normalization per channel
            normalized_segment = np.zeros_like(segment)
            for ch in range(segment.shape[0]):
                normalized_segment[ch, :] = zscore(segment[ch, :])
            normalized_segments.append(normalized_segment)
            
        return normalized_segments
    
    def fit_transform(self, eeg_data: np.ndarray, 
                     labels: Optional[np.ndarray] = None,
                     **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit the preprocessor and transform the data.
        
        Args:
            eeg_data: Raw EEG data
            labels: Optional labels for the data
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (preprocessed_data, labels)
        """
        preprocessed_data = self.preprocess_raw_data(eeg_data, **kwargs)
        self.is_fitted = True
        
        # Adjust labels if provided
        if labels is not None:
            n_segments = len(preprocessed_data)
            adjusted_labels = self._adjust_labels(labels, n_segments)
            return preprocessed_data, adjusted_labels
        
        return preprocessed_data, None
    
    def transform(self, eeg_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            eeg_data: Raw EEG data
            **kwargs: Additional arguments
            
        Returns:
            Preprocessed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform.")
        
        return self.preprocess_raw_data(eeg_data, **kwargs)
    
    def _adjust_labels(self, labels: np.ndarray, n_segments: int) -> np.ndarray:
        """
        Adjust labels to match the number of segments.
        
        Args:
            labels: Original labels
            n_segments: Number of segments
            
        Returns:
            Adjusted labels
        """
        if len(labels) == n_segments:
            return labels
        
        # Repeat labels for each segment from the same recording
        adjusted_labels = []
        for label in labels:
            adjusted_labels.extend([label] * n_segments)
        
        return np.array(adjusted_labels)
    
    def save_preprocessor(self, filepath: str):
        """Save the fitted preprocessor to disk."""
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath: str):
        """Load a fitted preprocessor from disk."""
        import joblib
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing pipeline."""
        return {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'scaler_fitted': hasattr(self.scaler, 'mean_'),
            'filter_info': self.signal_filter.get_filter_info(),
            'artifact_removal_info': self.artifact_remover.get_removal_info()
        } 