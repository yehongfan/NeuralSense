"""
Connectivity Features for EEG Data

This module provides connectivity feature extraction functionality for EEG signals,
including coherence, phase locking value, and other inter-channel measures.
"""

import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class ConnectivityFeatures:
    """
    Connectivity feature extractor for EEG signals.
    
    Extracts various connectivity features including:
    - Coherence between channels
    - Phase locking value
    - Correlation coefficients
    - Mutual information
    """
    
    def __init__(self):
        """Initialize the connectivity feature extractor."""
        pass
    
    def extract_all(self, eeg_data: np.ndarray, sampling_rate: int, 
                   feature_list: List[str]) -> Dict[str, float]:
        """
        Extract all specified connectivity features.
        
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
                if feature_name in ['coherence', 'phase_locking_value']:
                    features.update(method(eeg_data, sampling_rate))
                else:
                    features[feature_name] = method(eeg_data, sampling_rate)
            else:
                logger.warning(f"Unknown connectivity feature: {feature_name}")
        
        return features
    
    def extract_coherence(self, eeg_data: np.ndarray, sampling_rate: int) -> Dict[str, float]:
        """
        Extract coherence features between channel pairs.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with coherence features
        """
        features = {}
        n_channels = eeg_data.shape[0]
        
        # Calculate coherence for all channel pairs
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Calculate coherence
                freqs, coh = signal.coherence(eeg_data[i, :], eeg_data[j, :], 
                                            sampling_rate, nperseg=min(256, len(eeg_data[i, :])//4))
                
                if len(coh) > 0:
                    # Extract features from coherence
                    features[f'coherence_ch{i}_ch{j}_mean'] = np.mean(coh)
                    features[f'coherence_ch{i}_ch{j}_std'] = np.std(coh)
                    features[f'coherence_ch{i}_ch{j}_max'] = np.max(coh)
                    features[f'coherence_ch{i}_ch{j}_min'] = np.min(coh)
                    features[f'coherence_ch{i}_ch{j}_peak_freq'] = freqs[np.argmax(coh)]
        
        return features
    
    def extract_phase_locking_value(self, eeg_data: np.ndarray, sampling_rate: int) -> Dict[str, float]:
        """
        Extract phase locking value features between channel pairs.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with PLV features
        """
        features = {}
        n_channels = eeg_data.shape[0]
        
        # Calculate PLV for all channel pairs
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                plv = self._calculate_plv(eeg_data[i, :], eeg_data[j, :], sampling_rate)
                features[f'plv_ch{i}_ch{j}'] = plv
        
        return features
    
    def extract_correlation(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract average correlation coefficient across all channel pairs.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Average correlation coefficient
        """
        correlations = []
        n_channels = eeg_data.shape[0]
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                corr, _ = pearsonr(eeg_data[i, :], eeg_data[j, :])
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def extract_mutual_information(self, eeg_data: np.ndarray, sampling_rate: int) -> float:
        """
        Extract average mutual information across all channel pairs.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Average mutual information
        """
        mi_values = []
        n_channels = eeg_data.shape[0]
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                mi = self._calculate_mutual_information(eeg_data[i, :], eeg_data[j, :])
                mi_values.append(mi)
        
        return np.mean(mi_values) if mi_values else 0.0
    
    def extract_granger_causality(self, eeg_data: np.ndarray, sampling_rate: int) -> Dict[str, float]:
        """
        Extract Granger causality features between channel pairs.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with Granger causality features
        """
        features = {}
        n_channels = eeg_data.shape[0]
        
        # Calculate Granger causality for all channel pairs
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Calculate bidirectional Granger causality
                gc_ij = self._calculate_granger_causality(eeg_data[i, :], eeg_data[j, :])
                gc_ji = self._calculate_granger_causality(eeg_data[j, :], eeg_data[i, :])
                
                features[f'granger_ch{i}_to_ch{j}'] = gc_ij
                features[f'granger_ch{j}_to_ch{i}'] = gc_ji
                features[f'granger_ch{i}_ch{j}_net'] = gc_ij - gc_ji
        
        return features
    
    def extract_cross_correlation(self, eeg_data: np.ndarray, sampling_rate: int) -> Dict[str, float]:
        """
        Extract cross-correlation features between channel pairs.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with cross-correlation features
        """
        features = {}
        n_channels = eeg_data.shape[0]
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                # Calculate cross-correlation
                corr = signal.correlate(eeg_data[i, :], eeg_data[j, :], mode='full')
                
                # Extract features
                features[f'crosscorr_ch{i}_ch{j}_max'] = np.max(corr)
                features[f'crosscorr_ch{i}_ch{j}_min'] = np.min(corr)
                features[f'crosscorr_ch{i}_ch{j}_mean'] = np.mean(corr)
                features[f'crosscorr_ch{i}_ch{j}_std'] = np.std(corr)
                
                # Lag at maximum correlation
                max_lag = np.argmax(corr) - len(eeg_data[i, :]) + 1
                features[f'crosscorr_ch{i}_ch{j}_max_lag'] = max_lag
    
    def _calculate_plv(self, signal1: np.ndarray, signal2: np.ndarray, sampling_rate: int) -> float:
        """
        Calculate phase locking value between two signals.
        
        Args:
            signal1: First signal
            signal2: Second signal
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Phase locking value
        """
        # Calculate analytic signal using Hilbert transform
        analytic1 = signal.hilbert(signal1)
        analytic2 = signal.hilbert(signal2)
        
        # Extract phases
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # Calculate phase difference
        phase_diff = phase1 - phase2
        
        # Calculate PLV
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        
        return plv
    
    def _calculate_mutual_information(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """
        Calculate mutual information between two signals.
        
        Args:
            signal1: First signal
            signal2: Second signal
            
        Returns:
            Mutual information
        """
        # Discretize signals for MI calculation
        bins = min(20, len(signal1) // 100)
        if bins < 2:
            return 0.0
        
        # Create 2D histogram
        hist_2d, _, _ = np.histogram2d(signal1, signal2, bins=bins, density=True)
        hist_1d_1, _ = np.histogram(signal1, bins=bins, density=True)
        hist_1d_2, _ = np.histogram(signal2, bins=bins, density=True)
        
        # Calculate mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if hist_2d[i, j] > 0 and hist_1d_1[i] > 0 and hist_1d_2[j] > 0:
                    mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (hist_1d_1[i] * hist_1d_2[j]))
        
        return mi
    
    def _calculate_granger_causality(self, signal1: np.ndarray, signal2: np.ndarray, 
                                   max_lag: int = 10) -> float:
        """
        Calculate Granger causality from signal1 to signal2.
        
        Args:
            signal1: First signal (cause)
            signal2: Second signal (effect)
            max_lag: Maximum lag for AR model
            
        Returns:
            Granger causality value
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Prepare data for Granger causality test
            data = np.column_stack((signal2, signal1))
            
            # Perform Granger causality test
            gc_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            
            # Extract F-statistic from the last lag
            f_stat = gc_result[max_lag][0]['ssr_ftest'][0]
            
            return f_stat
            
        except ImportError:
            # Fallback to simplified implementation
            logger.warning("statsmodels not available, using simplified Granger causality")
            return self._simplified_granger_causality(signal1, signal2, max_lag)
    
    def _simplified_granger_causality(self, signal1: np.ndarray, signal2: np.ndarray, 
                                    max_lag: int = 10) -> float:
        """
        Simplified Granger causality calculation.
        
        Args:
            signal1: First signal (cause)
            signal2: Second signal (effect)
            max_lag: Maximum lag for AR model
            
        Returns:
            Simplified Granger causality value
        """
        # This is a simplified implementation
        # In practice, you should use statsmodels for proper Granger causality
        
        # Calculate correlation with different lags
        correlations = []
        for lag in range(1, min(max_lag + 1, len(signal1) // 4)):
            if lag < len(signal1):
                corr, _ = pearsonr(signal1[:-lag], signal2[lag:])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0 