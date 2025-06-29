"""
Feature Extractor for EEG Data

This module provides comprehensive feature extraction functionality for EEG data,
combining time-domain, frequency-domain, time-frequency, and connectivity features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import signal
from sklearn.preprocessing import StandardScaler

from .time_domain_features import TimeDomainFeatures
from .frequency_domain_features import FrequencyDomainFeatures
from .time_frequency_features import TimeFrequencyFeatures
from .connectivity_features import ConnectivityFeatures

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Comprehensive feature extractor for EEG data.
    
    This class extracts various types of features from EEG signals:
    - Time-domain features (statistical measures)
    - Frequency-domain features (spectral analysis)
    - Time-frequency features (wavelet analysis)
    - Connectivity features (inter-channel relationships)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration dictionary with feature extraction parameters
        """
        self.config = config or self._get_default_config()
        self.time_features = TimeDomainFeatures()
        self.freq_features = FrequencyDomainFeatures()
        self.tf_features = TimeFrequencyFeatures()
        self.conn_features = ConnectivityFeatures()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for feature extraction."""
        return {
            'time_domain': ['mean', 'std', 'variance', 'skewness', 'kurtosis', 
                           'zero_crossings', 'peak_to_peak', 'rms', 'entropy'],
            'frequency_domain': ['power_spectral_density', 'spectral_entropy', 
                               'spectral_centroid', 'spectral_bandwidth', 
                               'spectral_rolloff', 'spectral_flatness'],
            'time_frequency': ['wavelet_coefficients', 'spectrogram_features'],
            'connectivity': ['coherence', 'phase_locking_value'],
            'frequency_bands': {
                'delta': [0.5, 4],
                'theta': [4, 8],
                'alpha': [8, 13],
                'beta': [13, 30],
                'gamma': [30, 45]
            }
        }
    
    def extract_features(self, eeg_data: np.ndarray, sampling_rate: int) -> pd.DataFrame:
        """
        Extract comprehensive features from EEG data.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            DataFrame with extracted features
        """
        logger.info("Starting feature extraction...")
        
        features_dict = {}
        
        # Extract time-domain features
        if self.config.get('time_domain'):
            logger.info("Extracting time-domain features...")
            time_features = self.time_features.extract_all(eeg_data, 
                                                          self.config['time_domain'])
            features_dict.update(time_features)
        
        # Extract frequency-domain features
        if self.config.get('frequency_domain'):
            logger.info("Extracting frequency-domain features...")
            freq_features = self.freq_features.extract_all(eeg_data, sampling_rate,
                                                          self.config['frequency_domain'],
                                                          self.config.get('frequency_bands', {}))
            features_dict.update(freq_features)
        
        # Extract time-frequency features
        if self.config.get('time_frequency'):
            logger.info("Extracting time-frequency features...")
            tf_features = self.tf_features.extract_all(eeg_data, sampling_rate,
                                                      self.config['time_frequency'])
            features_dict.update(tf_features)
        
        # Extract connectivity features
        if self.config.get('connectivity'):
            logger.info("Extracting connectivity features...")
            conn_features = self.conn_features.extract_all(eeg_data, sampling_rate,
                                                          self.config['connectivity'])
            features_dict.update(conn_features)
        
        # Create feature DataFrame
        feature_df = pd.DataFrame([features_dict])
        
        logger.info(f"Extracted {len(feature_df.columns)} features")
        return feature_df
    
    def extract_features_batch(self, eeg_segments: List[np.ndarray], 
                             sampling_rate: int) -> pd.DataFrame:
        """
        Extract features from multiple EEG segments.
        
        Args:
            eeg_segments: List of EEG segments
            sampling_rate: Sampling rate in Hz
            
        Returns:
            DataFrame with features for all segments
        """
        logger.info(f"Extracting features from {len(eeg_segments)} segments...")
        
        all_features = []
        
        for i, segment in enumerate(eeg_segments):
            if i % 100 == 0:
                logger.info(f"Processing segment {i+1}/{len(eeg_segments)}")
            
            features = self.extract_features(segment, sampling_rate)
            all_features.append(features)
        
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        
        logger.info(f"Extracted features from {len(eeg_segments)} segments")
        return combined_features
    
    def fit_transform(self, eeg_segments: List[np.ndarray], 
                     sampling_rate: int) -> pd.DataFrame:
        """
        Fit the feature extractor and transform the data.
        
        Args:
            eeg_segments: List of EEG segments
            sampling_rate: Sampling rate in Hz
            
        Returns:
            DataFrame with extracted and scaled features
        """
        # Extract features
        features_df = self.extract_features_batch(eeg_segments, sampling_rate)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features_df)
        scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)
        
        self.is_fitted = True
        logger.info("Feature extraction and scaling completed")
        
        return scaled_df
    
    def transform(self, eeg_segments: List[np.ndarray], 
                 sampling_rate: int) -> pd.DataFrame:
        """
        Transform new data using fitted feature extractor.
        
        Args:
            eeg_segments: List of EEG segments
            sampling_rate: Sampling rate in Hz
            
        Returns:
            DataFrame with extracted and scaled features
        """
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform.")
        
        # Extract features
        features_df = self.extract_features_batch(eeg_segments, sampling_rate)
        
        # Scale features using fitted scaler
        scaled_features = self.scaler.transform(features_df)
        scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)
        
        return scaled_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted first.")
        
        return list(self.scaler.get_feature_names_out())
    
    def get_feature_importance(self, model) -> Dict[str, float]:
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute.")
        
        feature_names = self.get_feature_names()
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_feature_extractor(self, filepath: str):
        """Save the fitted feature extractor to disk."""
        import joblib
        joblib.dump(self, filepath)
        logger.info(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load_feature_extractor(cls, filepath: str):
        """Load a fitted feature extractor from disk."""
        import joblib
        extractor = joblib.load(filepath)
        logger.info(f"Feature extractor loaded from {filepath}")
        return extractor
    
    def get_extraction_info(self) -> Dict[str, Any]:
        """Get information about the feature extraction process."""
        return {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'scaler_fitted': hasattr(self.scaler, 'mean_'),
            'n_features': len(self.get_feature_names()) if self.is_fitted else 0
        } 