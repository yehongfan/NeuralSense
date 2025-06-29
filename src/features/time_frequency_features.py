"""
Time-Frequency Features for EEG Data

This module provides time-frequency feature extraction functionality for EEG signals,
including wavelet analysis and spectrogram features.
"""

import numpy as np
import pywt
from scipy import signal
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class TimeFrequencyFeatures:
    """
    Time-frequency feature extractor for EEG signals.
    
    Extracts various time-frequency features including:
    - Wavelet coefficients
    - Spectrogram features
    - Wavelet energy
    - Wavelet entropy
    """
    
    def __init__(self):
        """Initialize the time-frequency feature extractor."""
        pass
    
    def extract_all(self, eeg_data: np.ndarray, sampling_rate: int, 
                   feature_list: List[str]) -> Dict[str, float]:
        """
        Extract all specified time-frequency features.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            feature_list: List of feature names to extract
            
        Returns:
            Dictionary with feature names and values
        """
        features = {}
        
        for feature_name in feature_list:
            if hasattr(self, f'extract_{feature_name}'):
                method = getattr(self, f'extract_{feature_name}')
                if feature_name == 'wavelet_coefficients':
                    features.update(method(eeg_data, sampling_rate))
                elif feature_name == 'spectrogram_features':
                    features.update(method(eeg_data, sampling_rate))
                else:
                    features[feature_name] = method(eeg_data, sampling_rate)
            else:
                logger.warning(f"Unknown time-frequency feature: {feature_name}")
        
        return features
    
    def extract_wavelet_coefficients(self, eeg_data: np.ndarray, sampling_rate: int) -> Dict[str, float]:
        """
        Extract wavelet coefficient features.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with wavelet features
        """
        features = {}
        
        # Define wavelet and decomposition levels
        wavelet = 'db4'
        max_level = min(5, int(np.log2(eeg_data.shape[1])))
        
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal_ch, wavelet, level=max_level)
            
            # Extract features from each decomposition level
            for level, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    # Statistical features for each level
                    features[f'wavelet_ch{ch}_level{level}_mean'] = np.mean(coeff)
                    features[f'wavelet_ch{ch}_level{level}_std'] = np.std(coeff)
                    features[f'wavelet_ch{ch}_level{level}_energy'] = np.sum(coeff ** 2)
                    features[f'wavelet_ch{ch}_level{level}_entropy'] = self._calculate_entropy(coeff)
        
        return features
    
    def extract_spectrogram_features(self, eeg_data: np.ndarray, sampling_rate: int) -> Dict[str, float]:
        """
        Extract spectrogram features.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with spectrogram features
        """
        features = {}
        
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            
            # Calculate spectrogram
            freqs, times, Sxx = signal.spectrogram(signal_ch, sampling_rate, 
                                                  nperseg=min(256, len(signal_ch)//4),
                                                  noverlap=min(128, len(signal_ch)//8))
            
            if Sxx.size > 0:
                # Statistical features of spectrogram
                features[f'spectrogram_ch{ch}_mean'] = np.mean(Sxx)
                features[f'spectrogram_ch{ch}_std'] = np.std(Sxx)
                features[f'spectrogram_ch{ch}_max'] = np.max(Sxx)
                features[f'spectrogram_ch{ch}_min'] = np.min(Sxx)
                features[f'spectrogram_ch{ch}_energy'] = np.sum(Sxx)
                features[f'spectrogram_ch{ch}_entropy'] = self._calculate_entropy(Sxx.flatten())
                
                # Time-frequency features
                features[f'spectrogram_ch{ch}_peak_freq'] = freqs[np.argmax(np.mean(Sxx, axis=1))]
                features[f'spectrogram_ch{ch}_peak_time'] = times[np.argmax(np.mean(Sxx, axis=0))]
        
        return features
    
    def extract_wavelet_energy(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract total wavelet energy across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Total wavelet energy
        """
        total_energy = 0.0
        wavelet = 'db4'
        max_level = min(5, int(np.log2(eeg_data.shape[1])))
        
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            coeffs = pywt.wavedec(signal_ch, wavelet, level=max_level)
            
            for coeff in coeffs:
                total_energy += np.sum(coeff ** 2)
        
        return total_energy
    
    def extract_wavelet_entropy(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract wavelet entropy across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Wavelet entropy
        """
        entropies = []
        wavelet = 'db4'
        max_level = min(5, int(np.log2(eeg_data.shape[1])))
        
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            coeffs = pywt.wavedec(signal_ch, wavelet, level=max_level)
            
            # Calculate energy distribution across levels
            energies = [np.sum(coeff ** 2) for coeff in coeffs]
            total_energy = np.sum(energies)
            
            if total_energy > 0:
                # Normalize energies
                normalized_energies = [e / total_energy for e in energies]
                normalized_energies = [e for e in normalized_energies if e > 0]
                
                if len(normalized_energies) > 0:
                    entropy = -np.sum(normalized_energies * np.log2(normalized_energies))
                    entropies.append(entropy)
        
        return np.mean(entropies) if entropies else 0.0
    
    def extract_wavelet_variance(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract wavelet variance across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Wavelet variance
        """
        variances = []
        wavelet = 'db4'
        max_level = min(5, int(np.log2(eeg_data.shape[1])))
        
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            coeffs = pywt.wavedec(signal_ch, wavelet, level=max_level)
            
            for coeff in coeffs:
                variances.append(np.var(coeff))
        
        return np.mean(variances) if variances else 0.0
    
    def extract_wavelet_skewness(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract wavelet skewness across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Wavelet skewness
        """
        skewnesses = []
        wavelet = 'db4'
        max_level = min(5, int(np.log2(eeg_data.shape[1])))
        
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            coeffs = pywt.wavedec(signal_ch, wavelet, level=max_level)
            
            for coeff in coeffs:
                if len(coeff) > 2:
                    skewness = self._calculate_skewness(coeff)
                    skewnesses.append(skewness)
        
        return np.mean(skewnesses) if skewnesses else 0.0
    
    def extract_wavelet_kurtosis(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract wavelet kurtosis across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Wavelet kurtosis
        """
        kurtoses = []
        wavelet = 'db4'
        max_level = min(5, int(np.log2(eeg_data.shape[1])))
        
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            coeffs = pywt.wavedec(signal_ch, wavelet, level=max_level)
            
            for coeff in coeffs:
                if len(coeff) > 3:
                    kurtosis = self._calculate_kurtosis(coeff)
                    kurtoses.append(kurtosis)
        
        return np.mean(kurtoses) if kurtoses else 0.0
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data."""
        if len(data) == 0:
            return 0.0
        
        # Discretize data for entropy calculation
        hist, _ = np.histogram(data, bins=min(50, len(data)//10), density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        
        if len(hist) == 0:
            return 0.0
        
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3 