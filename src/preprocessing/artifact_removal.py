"""
Artifact Removal Module for EEG Data

This module provides methods for detecting and removing various artifacts
from EEG signals, including eye blinks, muscle activity, and line noise.
"""

import numpy as np
import mne
from scipy import signal
from scipy.stats import zscore
from sklearn.decomposition import FastICA
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ArtifactRemover:
    """
    Artifact removal class for EEG data preprocessing.
    
    Provides methods for:
    - Eye blink detection and removal
    - Muscle artifact detection
    - Line noise removal
    - ICA-based artifact removal
    - Threshold-based artifact detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the artifact remover.
        
        Args:
            config: Configuration dictionary with artifact removal parameters
        """
        self.config = config or self._get_default_config()
        self.artifacts_removed = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for artifact removal."""
        return {
            'eye_blink_threshold': 3.0,
            'muscle_threshold': 2.5,
            'line_noise_threshold': 2.0,
            'amplitude_threshold': 100,  # microvolts
            'variance_threshold': 3.0,
            'use_ica': True,
            'ica_components': 10
        }
    
    def remove_artifacts(self, eeg_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Apply comprehensive artifact removal to EEG data.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Clean EEG data with artifacts removed
        """
        clean_data = eeg_data.copy()
        
        # Step 1: Remove line noise
        if self.config.get('remove_line_noise', True):
            clean_data = self.remove_line_noise(clean_data, sampling_rate)
            self.artifacts_removed.append('line_noise')
        
        # Step 2: Remove amplitude outliers
        if self.config.get('remove_amplitude_outliers', True):
            clean_data = self.remove_amplitude_outliers(clean_data)
            self.artifacts_removed.append('amplitude_outliers')
        
        # Step 3: Remove variance outliers
        if self.config.get('remove_variance_outliers', True):
            clean_data = self.remove_variance_outliers(clean_data)
            self.artifacts_removed.append('variance_outliers')
        
        # Step 4: ICA-based artifact removal
        if self.config.get('use_ica', True):
            clean_data = self.remove_artifacts_ica(clean_data, sampling_rate)
            self.artifacts_removed.append('ica_artifacts')
        
        logger.info(f"Removed artifacts: {self.artifacts_removed}")
        return clean_data
    
    def remove_line_noise(self, eeg_data: np.ndarray, sampling_rate: int,
                         line_freq: float = 50.0) -> np.ndarray:
        """
        Remove line noise (power line interference) using notch filtering.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            line_freq: Line frequency (50 or 60 Hz)
            
        Returns:
            EEG data with line noise removed
        """
        # Design notch filter for line frequency
        nyquist = sampling_rate / 2
        notch_freq = line_freq / nyquist
        b, a = signal.iirnotch(line_freq, 30.0, sampling_rate)
        
        # Apply filter to each channel
        clean_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            clean_data[ch, :] = signal.filtfilt(b, a, eeg_data[ch, :])
        
        logger.info(f"Removed line noise at {line_freq} Hz")
        return clean_data
    
    def remove_amplitude_outliers(self, eeg_data: np.ndarray,
                                threshold: Optional[float] = None) -> np.ndarray:
        """
        Remove segments with amplitude outliers.
        
        Args:
            eeg_data: EEG data (channels x samples)
            threshold: Amplitude threshold in microvolts
            
        Returns:
            EEG data with amplitude outliers removed
        """
        if threshold is None:
            threshold = self.config['amplitude_threshold']
        
        # Calculate amplitude for each channel
        amplitudes = np.max(np.abs(eeg_data), axis=1)
        
        # Find channels with excessive amplitude
        outlier_channels = amplitudes > threshold
        
        if np.any(outlier_channels):
            # Interpolate or zero out outlier channels
            clean_data = eeg_data.copy()
            for ch in np.where(outlier_channels)[0]:
                clean_data[ch, :] = np.median(eeg_data[ch, :])
            
            logger.info(f"Removed amplitude outliers from {np.sum(outlier_channels)} channels")
            return clean_data
        
        return eeg_data
    
    def remove_variance_outliers(self, eeg_data: np.ndarray,
                               threshold: Optional[float] = None) -> np.ndarray:
        """
        Remove segments with variance outliers.
        
        Args:
            eeg_data: EEG data (channels x samples)
            threshold: Variance threshold (z-score)
            
        Returns:
            EEG data with variance outliers removed
        """
        if threshold is None:
            threshold = self.config['variance_threshold']
        
        # Calculate variance for each channel
        variances = np.var(eeg_data, axis=1)
        z_scores = zscore(variances)
        
        # Find channels with excessive variance
        outlier_channels = np.abs(z_scores) > threshold
        
        if np.any(outlier_channels):
            # Interpolate or zero out outlier channels
            clean_data = eeg_data.copy()
            for ch in np.where(outlier_channels)[0]:
                clean_data[ch, :] = np.median(eeg_data[ch, :])
            
            logger.info(f"Removed variance outliers from {np.sum(outlier_channels)} channels")
            return clean_data
        
        return eeg_data
    
    def remove_artifacts_ica(self, eeg_data: np.ndarray, sampling_rate: int,
                           n_components: Optional[int] = None) -> np.ndarray:
        """
        Remove artifacts using Independent Component Analysis (ICA).
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            n_components: Number of ICA components
            
        Returns:
            EEG data with ICA artifacts removed
        """
        if n_components is None:
            n_components = min(self.config['ica_components'], eeg_data.shape[0])
        
        # Transpose data for ICA (samples x channels)
        data_t = eeg_data.T
        
        # Apply ICA
        ica = FastICA(n_components=n_components, random_state=42)
        ica_components = ica.fit_transform(data_t)
        
        # Identify artifact components (simplified approach)
        artifact_components = self._identify_artifact_components(ica_components, sampling_rate)
        
        # Reconstruct signal without artifact components
        mixing_matrix = ica.mixing_
        unmixing_matrix = ica.components_
        
        # Zero out artifact components
        clean_components = ica_components.copy()
        clean_components[:, artifact_components] = 0
        
        # Reconstruct signal
        clean_data_t = np.dot(clean_components, unmixing_matrix)
        clean_data = clean_data_t.T
        
        logger.info(f"Removed {len(artifact_components)} artifact components using ICA")
        return clean_data
    
    def _identify_artifact_components(self, ica_components: np.ndarray,
                                    sampling_rate: int) -> List[int]:
        """
        Identify artifact components based on various criteria.
        
        Args:
            ica_components: ICA components (samples x components)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            List of artifact component indices
        """
        artifact_components = []
        
        for i in range(ica_components.shape[1]):
            component = ica_components[:, i]
            
            # Calculate various metrics
            kurtosis = self._calculate_kurtosis(component)
            variance = np.var(component)
            spectral_entropy = self._calculate_spectral_entropy(component, sampling_rate)
            
            # Simple heuristic for artifact detection
            if (kurtosis > 5.0 or  # High kurtosis (spiky artifacts)
                variance > np.percentile([np.var(ica_components[:, j]) for j in range(ica_components.shape[1])], 95) or  # High variance
                spectral_entropy < 0.5):  # Low spectral entropy (line noise)
                artifact_components.append(i)
        
        return artifact_components
    
    def _calculate_kurtosis(self, signal_data: np.ndarray) -> float:
        """Calculate kurtosis of a signal."""
        from scipy.stats import kurtosis
        return kurtosis(signal_data)
    
    def _calculate_spectral_entropy(self, signal_data: np.ndarray, sampling_rate: int) -> float:
        """Calculate spectral entropy of a signal."""
        # Compute power spectral density
        freqs, psd = signal.welch(signal_data, sampling_rate)
        
        # Normalize PSD
        psd_norm = psd / np.sum(psd)
        
        # Calculate spectral entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        return entropy
    
    def detect_eye_blinks(self, eeg_data: np.ndarray, 
                         frontal_channels: List[int] = None) -> np.ndarray:
        """
        Detect eye blink artifacts in frontal channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            frontal_channels: Indices of frontal channels
            
        Returns:
            Boolean array indicating eye blink locations
        """
        if frontal_channels is None:
            # Default frontal channels (Fp1, Fp2, F7, F3, Fz, F4, F8)
            frontal_channels = [0, 1, 2, 3, 4, 5, 6]
        
        # Calculate amplitude in frontal channels
        frontal_data = eeg_data[frontal_channels, :]
        amplitude = np.max(np.abs(frontal_data), axis=0)
        
        # Detect blinks using threshold
        threshold = self.config['eye_blink_threshold'] * np.std(amplitude)
        blink_mask = amplitude > threshold
        
        logger.info(f"Detected {np.sum(blink_mask)} potential eye blinks")
        return blink_mask
    
    def detect_muscle_artifacts(self, eeg_data: np.ndarray,
                              temporal_channels: List[int] = None) -> np.ndarray:
        """
        Detect muscle artifacts in temporal channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            temporal_channels: Indices of temporal channels
            
        Returns:
            Boolean array indicating muscle artifact locations
        """
        if temporal_channels is None:
            # Default temporal channels (T3, T4, T5, T6)
            temporal_channels = [7, 11, 12, 16]
        
        # Calculate high-frequency power in temporal channels
        temporal_data = eeg_data[temporal_channels, :]
        
        # Apply high-pass filter to focus on muscle activity
        nyquist = 250 / 2  # Assuming 250 Hz sampling rate
        high_freq = 20 / nyquist
        b, a = signal.butter(4, high_freq, btype='high')
        
        filtered_data = np.zeros_like(temporal_data)
        for ch in range(temporal_data.shape[0]):
            filtered_data[ch, :] = signal.filtfilt(b, a, temporal_data[ch, :])
        
        # Calculate power
        power = np.mean(filtered_data ** 2, axis=0)
        
        # Detect muscle artifacts using threshold
        threshold = self.config['muscle_threshold'] * np.std(power)
        muscle_mask = power > threshold
        
        logger.info(f"Detected {np.sum(muscle_mask)} potential muscle artifacts")
        return muscle_mask
    
    def get_removal_info(self) -> Dict[str, Any]:
        """Get information about artifact removal process."""
        return {
            'artifacts_removed': self.artifacts_removed,
            'config': self.config
        } 