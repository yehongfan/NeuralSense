"""
Frequency-Domain Features for EEG Data

This module provides frequency-domain feature extraction functionality for EEG signals,
including spectral analysis, power spectral density, and frequency band features.
"""

import numpy as np
from scipy import signal
from scipy.stats import entropy
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class FrequencyDomainFeatures:
    """
    Frequency-domain feature extractor for EEG signals.
    
    Extracts various spectral features including:
    - Power spectral density
    - Spectral entropy
    - Spectral moments (centroid, bandwidth, rolloff)
    - Frequency band powers
    - Spectral flatness
    """
    
    def __init__(self):
        """Initialize the frequency-domain feature extractor."""
        pass
    
    def extract_all(self, eeg_data: np.ndarray, sampling_rate: int, 
                   feature_list: List[str], frequency_bands: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Extract all specified frequency-domain features.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            feature_list: List of feature names to extract
            frequency_bands: Dictionary of frequency band definitions
            
        Returns:
            Dictionary with feature names and values
        """
        features = {}
        
        for feature_name in feature_list:
            if hasattr(self, f'extract_{feature_name}'):
                method = getattr(self, f'extract_{feature_name}')
                if feature_name == 'power_spectral_density':
                    features.update(method(eeg_data, sampling_rate, frequency_bands))
                else:
                    features[feature_name] = method(eeg_data, sampling_rate)
            else:
                logger.warning(f"Unknown frequency-domain feature: {feature_name}")
        
        return features
    
    def extract_power_spectral_density(self, eeg_data: np.ndarray, sampling_rate: int,
                                     frequency_bands: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Extract power spectral density features for different frequency bands.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            frequency_bands: Dictionary of frequency band definitions
            
        Returns:
            Dictionary with power features for each frequency band
        """
        features = {}
        
        # Calculate PSD for each channel
        psd_features = []
        for ch in range(eeg_data.shape[0]):
            freqs, psd = signal.welch(eeg_data[ch, :], sampling_rate, nperseg=min(256, len(eeg_data[ch, :])))
            psd_features.append((freqs, psd))
        
        # Extract features for each frequency band
        for band_name, (low_freq, high_freq) in frequency_bands.items():
            band_powers = []
            for freqs, psd in psd_features:
                # Find frequency indices within the band
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(band_mask):
                    band_power = np.mean(psd[band_mask])
                    band_powers.append(band_power)
            
            if band_powers:
                features[f'psd_{band_name}_mean'] = np.mean(band_powers)
                features[f'psd_{band_name}_std'] = np.std(band_powers)
                features[f'psd_{band_name}_max'] = np.max(band_powers)
                features[f'psd_{band_name}_min'] = np.min(band_powers)
        
        return features
    
    def extract_spectral_entropy(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract spectral entropy across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Spectral entropy value
        """
        spectral_entropies = []
        
        for ch in range(eeg_data.shape[0]):
            freqs, psd = signal.welch(eeg_data[ch, :], sampling_rate, nperseg=min(256, len(eeg_data[ch, :])))
            
            # Normalize PSD
            psd_norm = psd / np.sum(psd)
            psd_norm = psd_norm[psd_norm > 0]  # Remove zero probabilities
            
            if len(psd_norm) > 0:
                entropy_val = -np.sum(psd_norm * np.log2(psd_norm))
                spectral_entropies.append(entropy_val)
        
        return np.mean(spectral_entropies) if spectral_entropies else 0.0
    
    def extract_spectral_centroid(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract spectral centroid across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Spectral centroid value
        """
        centroids = []
        
        for ch in range(eeg_data.shape[0]):
            freqs, psd = signal.welch(eeg_data[ch, :], sampling_rate, nperseg=min(256, len(eeg_data[ch, :])))
            
            # Calculate centroid
            if np.sum(psd) > 0:
                centroid = np.sum(freqs * psd) / np.sum(psd)
                centroids.append(centroid)
        
        return np.mean(centroids) if centroids else 0.0
    
    def extract_spectral_bandwidth(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract spectral bandwidth across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Spectral bandwidth value
        """
        bandwidths = []
        
        for ch in range(eeg_data.shape[0]):
            freqs, psd = signal.welch(eeg_data[ch, :], sampling_rate, nperseg=min(256, len(eeg_data[ch, :])))
            
            # Calculate centroid first
            if np.sum(psd) > 0:
                centroid = np.sum(freqs * psd) / np.sum(psd)
                
                # Calculate bandwidth
                bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd))
                bandwidths.append(bandwidth)
        
        return np.mean(bandwidths) if bandwidths else 0.0
    
    def extract_spectral_rolloff(self, eeg_data: np.ndarray, sampling_rate: int, 
                               rolloff_percent: float = 0.85) -> float:
        """
        Extract spectral rolloff across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            rolloff_percent: Rolloff percentage (default: 0.85)
            
        Returns:
            Spectral rolloff value
        """
        rolloffs = []
        
        for ch in range(eeg_data.shape[0]):
            freqs, psd = signal.welch(eeg_data[ch, :], sampling_rate, nperseg=min(256, len(eeg_data[ch, :])))
            
            # Calculate cumulative sum
            cumsum = np.cumsum(psd)
            threshold = rolloff_percent * cumsum[-1]
            
            # Find rolloff frequency
            rolloff_idx = np.where(cumsum >= threshold)[0]
            if len(rolloff_idx) > 0:
                rolloff_freq = freqs[rolloff_idx[0]]
                rolloffs.append(rolloff_freq)
        
        return np.mean(rolloffs) if rolloffs else 0.0
    
    def extract_spectral_flatness(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract spectral flatness across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Spectral flatness value
        """
        flatnesses = []
        
        for ch in range(eeg_data.shape[0]):
            freqs, psd = signal.welch(eeg_data[ch, :], sampling_rate, nperseg=min(256, len(eeg_data[ch, :])))
            
            # Remove zero values
            psd_positive = psd[psd > 0]
            
            if len(psd_positive) > 0:
                # Calculate geometric and arithmetic means
                geometric_mean = np.exp(np.mean(np.log(psd_positive)))
                arithmetic_mean = np.mean(psd_positive)
                
                flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0
                flatnesses.append(flatness)
        
        return np.mean(flatnesses) if flatnesses else 0.0
    
    def extract_spectral_contrast(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract spectral contrast across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Spectral contrast value
        """
        contrasts = []
        
        for ch in range(eeg_data.shape[0]):
            freqs, psd = signal.welch(eeg_data[ch, :], sampling_rate, nperseg=min(256, len(eeg_data[ch, :])))
            
            if len(psd) > 0:
                contrast = (np.max(psd) - np.min(psd)) / (np.max(psd) + np.min(psd) + 1e-10)
                contrasts.append(contrast)
        
        return np.mean(contrasts) if contrasts else 0.0
    
    def extract_spectral_flux(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract spectral flux across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Spectral flux value
        """
        fluxes = []
        
        for ch in range(eeg_data.shape[0]):
            signal_ch = eeg_data[ch, :]
            
            # Split signal into segments
            segment_length = min(256, len(signal_ch) // 4)
            if segment_length < 64:
                continue
                
            segments = [signal_ch[i:i+segment_length] for i in range(0, len(signal_ch)-segment_length, segment_length//2)]
            
            if len(segments) < 2:
                continue
            
            # Calculate PSD for each segment
            segment_psds = []
            for segment in segments:
                freqs, psd = signal.welch(segment, sampling_rate, nperseg=min(128, len(segment)))
                segment_psds.append(psd)
            
            # Calculate flux between consecutive segments
            flux_values = []
            for i in range(1, len(segment_psds)):
                flux = np.sum((segment_psds[i] - segment_psds[i-1]) ** 2)
                flux_values.append(flux)
            
            if flux_values:
                fluxes.append(np.mean(flux_values))
        
        return np.mean(fluxes) if fluxes else 0.0
    
    def extract_spectral_irregularity(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract spectral irregularity across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Spectral irregularity value
        """
        irregularities = []
        
        for ch in range(eeg_data.shape[0]):
            freqs, psd = signal.welch(eeg_data[ch, :], sampling_rate, nperseg=min(256, len(eeg_data[ch, :])))
            
            if len(psd) > 2:
                # Calculate irregularity as sum of squared differences between consecutive bins
                irregularity = np.sum(np.diff(psd) ** 2)
                irregularities.append(irregularity)
        
        return np.mean(irregularities) if irregularities else 0.0
    
    def extract_spectral_decrease(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract spectral decrease across all channels.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Spectral decrease value
        """
        decreases = []
        
        for ch in range(eeg_data.shape[0]):
            freqs, psd = signal.welch(eeg_data[ch, :], sampling_rate, nperseg=min(256, len(eeg_data[ch, :])))
            
            if len(psd) > 1:
                # Calculate spectral decrease
                decrease = np.sum((psd[1:] - psd[0]) / np.arange(1, len(psd))) / np.sum(psd[1:])
                decreases.append(decrease)
        
        return np.mean(decreases) if decreases else 0.0 