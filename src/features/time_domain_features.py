"""
Time-Domain Features for EEG Data

This module provides time-domain feature extraction functionality for EEG signals,
including statistical measures, signal characteristics, and temporal properties.
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class TimeDomainFeatures:
    """
    Time-domain feature extractor for EEG signals.
    
    Extracts various statistical and temporal features including:
    - Basic statistics (mean, std, variance, etc.)
    - Signal characteristics (peaks, crossings, etc.)
    - Distribution properties (skewness, kurtosis)
    - Entropy measures
    """
    
    def __init__(self):
        """Initialize the time-domain feature extractor."""
        pass
    
    def extract_all(self, eeg_data: np.ndarray, feature_list: List[str]) -> Dict[str, float]:
        """
        Extract all specified time-domain features.
        
        Args:
            eeg_data: EEG data (channels x samples)
            feature_list: List of feature names to extract
            
        Returns:
            Dictionary with feature names and values
        """
        features = {}
        
        for feature_name in feature_list:
            if hasattr(self, f'extract_{feature_name}'):
                method = getattr(self, f'extract_{feature_name}')
                features[feature_name] = method(eeg_data)
            else:
                logger.warning(f"Unknown time-domain feature: {feature_name}")
        
        return features
    
    def extract_mean(self, eeg_data: np.ndarray) -> float:
        """Extract mean value across all channels."""
        return np.mean(eeg_data)
    
    def extract_std(self, eeg_data: np.ndarray) -> float:
        """Extract standard deviation across all channels."""
        return np.std(eeg_data)
    
    def extract_variance(self, eeg_data: np.ndarray) -> float:
        """Extract variance across all channels."""
        return np.var(eeg_data)
    
    def extract_skewness(self, eeg_data: np.ndarray) -> float:
        """Extract skewness across all channels."""
        return stats.skew(eeg_data.flatten())
    
    def extract_kurtosis(self, eeg_data: np.ndarray) -> float:
        """Extract kurtosis across all channels."""
        return stats.kurtosis(eeg_data.flatten())
    
    def extract_zero_crossings(self, eeg_data: np.ndarray) -> float:
        """Extract number of zero crossings across all channels."""
        zero_crossings = 0
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            zero_crossings += np.sum(np.diff(np.signbit(signal_ch - np.mean(signal_ch))))
        
        return zero_crossings / eeg_data.shape[0]  # Average per channel
    
    def extract_peak_to_peak(self, eeg_data: np.ndarray) -> float:
        """Extract peak-to-peak amplitude across all channels."""
        return np.max(eeg_data) - np.min(eeg_data)
    
    def extract_rms(self, eeg_data: np.ndarray) -> float:
        """Extract root mean square across all channels."""
        return np.sqrt(np.mean(eeg_data ** 2))
    
    def extract_entropy(self, eeg_data: np.ndarray) -> float:
        """Extract Shannon entropy across all channels."""
        # Discretize the signal for entropy calculation
        hist, _ = np.histogram(eeg_data.flatten(), bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        return -np.sum(hist * np.log2(hist))
    
    def extract_median(self, eeg_data: np.ndarray) -> float:
        """Extract median value across all channels."""
        return np.median(eeg_data)
    
    def extract_range(self, eeg_data: np.ndarray) -> float:
        """Extract range (max - min) across all channels."""
        return np.ptp(eeg_data)
    
    def extract_peak_count(self, eeg_data: np.ndarray) -> float:
        """Extract number of peaks across all channels."""
        total_peaks = 0
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            peaks, _ = find_peaks(signal_ch, height=np.std(signal_ch))
            total_peaks += len(peaks)
        
        return total_peaks / eeg_data.shape[0]  # Average per channel
    
    def extract_peak_amplitude(self, eeg_data: np.ndarray) -> float:
        """Extract average peak amplitude across all channels."""
        peak_amplitudes = []
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            peaks, properties = find_peaks(signal_ch, height=np.std(signal_ch))
            if len(peaks) > 0:
                peak_amplitudes.extend(properties['peak_heights'])
        
        return np.mean(peak_amplitudes) if peak_amplitudes else 0.0
    
    def extract_peak_distance(self, eeg_data: np.ndarray) -> float:
        """Extract average distance between peaks across all channels."""
        peak_distances = []
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            peaks, _ = find_peaks(signal_ch, height=np.std(signal_ch))
            if len(peaks) > 1:
                distances = np.diff(peaks)
                peak_distances.extend(distances)
        
        return np.mean(peak_distances) if peak_distances else 0.0
    
    def extract_autocorrelation(self, eeg_data: np.ndarray, lag: int = 1) -> float:
        """Extract autocorrelation at specified lag across all channels."""
        autocorr_values = []
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            if len(signal_ch) > lag:
                autocorr = np.corrcoef(signal_ch[:-lag], signal_ch[lag:])[0, 1]
                if not np.isnan(autocorr):
                    autocorr_values.append(autocorr)
        
        return np.mean(autocorr_values) if autocorr_values else 0.0
    
    def extract_energy(self, eeg_data: np.ndarray) -> float:
        """Extract signal energy across all channels."""
        return np.sum(eeg_data ** 2)
    
    def extract_power(self, eeg_data: np.ndarray) -> float:
        """Extract signal power across all channels."""
        return np.mean(eeg_data ** 2)
    
    def extract_crest_factor(self, eeg_data: np.ndarray) -> float:
        """Extract crest factor (peak-to-RMS ratio) across all channels."""
        rms = self.extract_rms(eeg_data)
        peak = np.max(np.abs(eeg_data))
        return peak / rms if rms > 0 else 0.0
    
    def extract_impulse_factor(self, eeg_data: np.ndarray) -> float:
        """Extract impulse factor across all channels."""
        peak = np.max(np.abs(eeg_data))
        mean_abs = np.mean(np.abs(eeg_data))
        return peak / mean_abs if mean_abs > 0 else 0.0
    
    def extract_margin_factor(self, eeg_data: np.ndarray) -> float:
        """Extract margin factor across all channels."""
        peak = np.max(np.abs(eeg_data))
        rms = self.extract_rms(eeg_data)
        return peak / (rms ** 2) if rms > 0 else 0.0
    
    def extract_shape_factor(self, eeg_data: np.ndarray) -> float:
        """Extract shape factor across all channels."""
        rms = self.extract_rms(eeg_data)
        mean_abs = np.mean(np.abs(eeg_data))
        return rms / mean_abs if mean_abs > 0 else 0.0
    
    def extract_clearance_factor(self, eeg_data: np.ndarray) -> float:
        """Extract clearance factor across all channels."""
        peak = np.max(np.abs(eeg_data))
        mean_sqrt = np.mean(np.sqrt(np.abs(eeg_data)))
        return peak / (mean_sqrt ** 2) if mean_sqrt > 0 else 0.0
    
    def extract_percentile_25(self, eeg_data: np.ndarray) -> float:
        """Extract 25th percentile across all channels."""
        return np.percentile(eeg_data, 25)
    
    def extract_percentile_75(self, eeg_data: np.ndarray) -> float:
        """Extract 75th percentile across all channels."""
        return np.percentile(eeg_data, 75)
    
    def extract_iqr(self, eeg_data: np.ndarray) -> float:
        """Extract interquartile range across all channels."""
        return np.percentile(eeg_data, 75) - np.percentile(eeg_data, 25)
    
    def extract_mad(self, eeg_data: np.ndarray) -> float:
        """Extract median absolute deviation across all channels."""
        median = np.median(eeg_data)
        return np.median(np.abs(eeg_data - median))
    
    def extract_cv(self, eeg_data: np.ndarray) -> float:
        """Extract coefficient of variation across all channels."""
        mean = np.mean(eeg_data)
        std = np.std(eeg_data)
        return std / mean if mean != 0 else 0.0 