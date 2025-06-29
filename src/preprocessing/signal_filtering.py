"""
Signal Filtering Module for EEG Data

This module provides various filtering techniques for EEG signal preprocessing,
including notch filters, bandpass filters, and other signal enhancement methods.
"""

import numpy as np
import mne
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SignalFilter:
    """
    Signal filtering class for EEG data preprocessing.
    
    Provides methods for:
    - Notch filtering (power line interference)
    - Bandpass filtering (frequency band selection)
    - High-pass and low-pass filtering
    - Butterworth filter design
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the signal filter.
        
        Args:
            config: Configuration dictionary with filter parameters
        """
        self.config = config
        self.filters_applied = []
        
    def apply_filters(self, eeg_data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Apply all configured filters to EEG data.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Filtered EEG data
        """
        filtered_data = eeg_data.copy()
        
        # Apply notch filter for power line interference
        if 'notch_filter' in self.config:
            notch_freq = self.config['notch_filter']
            filtered_data = self.apply_notch_filter(filtered_data, sampling_rate, notch_freq)
            self.filters_applied.append(f"notch_{notch_freq}Hz")
        
        # Apply bandpass filter
        if 'bandpass_filter' in self.config:
            low_freq, high_freq = self.config['bandpass_filter']
            filtered_data = self.apply_bandpass_filter(filtered_data, sampling_rate, low_freq, high_freq)
            self.filters_applied.append(f"bandpass_{low_freq}-{high_freq}Hz")
        
        logger.info(f"Applied filters: {self.filters_applied}")
        return filtered_data
    
    def apply_notch_filter(self, eeg_data: np.ndarray, sampling_rate: int, 
                          notch_freq: float, quality_factor: float = 30.0) -> np.ndarray:
        """
        Apply notch filter to remove power line interference.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            notch_freq: Frequency to notch out (typically 50 or 60 Hz)
            quality_factor: Quality factor of the filter
            
        Returns:
            Notch-filtered EEG data
        """
        # Design notch filter
        b, a = iirnotch(notch_freq, quality_factor, sampling_rate)
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_data[ch, :] = filtfilt(b, a, eeg_data[ch, :])
        
        logger.info(f"Applied notch filter at {notch_freq} Hz")
        return filtered_data
    
    def apply_bandpass_filter(self, eeg_data: np.ndarray, sampling_rate: int,
                            low_freq: float, high_freq: float, order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to select specific frequency range.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            low_freq: Lower cutoff frequency
            high_freq: Upper cutoff frequency
            order: Filter order
            
        Returns:
            Bandpass-filtered EEG data
        """
        # Design bandpass filter
        nyquist = sampling_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        b, a = butter(order, [low_norm, high_norm], btype='band')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_data[ch, :] = filtfilt(b, a, eeg_data[ch, :])
        
        logger.info(f"Applied bandpass filter: {low_freq}-{high_freq} Hz")
        return filtered_data
    
    def apply_highpass_filter(self, eeg_data: np.ndarray, sampling_rate: int,
                            cutoff_freq: float, order: int = 4) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency artifacts.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            cutoff_freq: Cutoff frequency
            order: Filter order
            
        Returns:
            High-pass filtered EEG data
        """
        nyquist = sampling_rate / 2
        cutoff_norm = cutoff_freq / nyquist
        
        b, a = butter(order, cutoff_norm, btype='high')
        
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_data[ch, :] = filtfilt(b, a, eeg_data[ch, :])
        
        logger.info(f"Applied high-pass filter: {cutoff_freq} Hz")
        return filtered_data
    
    def apply_lowpass_filter(self, eeg_data: np.ndarray, sampling_rate: int,
                           cutoff_freq: float, order: int = 4) -> np.ndarray:
        """
        Apply low-pass filter to remove high-frequency noise.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            cutoff_freq: Cutoff frequency
            order: Filter order
            
        Returns:
            Low-pass filtered EEG data
        """
        nyquist = sampling_rate / 2
        cutoff_norm = cutoff_freq / nyquist
        
        b, a = butter(order, cutoff_norm, btype='low')
        
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_data[ch, :] = filtfilt(b, a, eeg_data[ch, :])
        
        logger.info(f"Applied low-pass filter: {cutoff_freq} Hz")
        return filtered_data
    
    def apply_butterworth_filter(self, eeg_data: np.ndarray, sampling_rate: int,
                               cutoff_freqs: Tuple[float, float], 
                               filter_type: str = 'bandpass', order: int = 4) -> np.ndarray:
        """
        Apply Butterworth filter with specified parameters.
        
        Args:
            eeg_data: EEG data (channels x samples)
            sampling_rate: Sampling rate in Hz
            cutoff_freqs: Cutoff frequencies (single value for low/high pass, tuple for bandpass)
            filter_type: Type of filter ('low', 'high', 'bandpass', 'bandstop')
            order: Filter order
            
        Returns:
            Filtered EEG data
        """
        nyquist = sampling_rate / 2
        
        if filter_type in ['low', 'high']:
            cutoff_norm = cutoff_freqs / nyquist
            b, a = butter(order, cutoff_norm, btype=filter_type)
        else:
            cutoff_norm = [f / nyquist for f in cutoff_freqs]
            b, a = butter(order, cutoff_norm, btype=filter_type)
        
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_data[ch, :] = filtfilt(b, a, eeg_data[ch, :])
        
        logger.info(f"Applied Butterworth {filter_type} filter")
        return filtered_data
    
    def get_filter_info(self) -> Dict[str, Any]:
        """Get information about applied filters."""
        return {
            'filters_applied': self.filters_applied,
            'config': self.config
        }
    
    def plot_frequency_response(self, sampling_rate: int, filter_type: str = 'bandpass',
                              **kwargs) -> None:
        """
        Plot the frequency response of the filter.
        
        Args:
            sampling_rate: Sampling rate in Hz
            filter_type: Type of filter to plot
            **kwargs: Additional filter parameters
        """
        import matplotlib.pyplot as plt
        
        if filter_type == 'bandpass':
            low_freq, high_freq = self.config['bandpass_filter']
            nyquist = sampling_rate / 2
            low_norm = low_freq / nyquist
            high_norm = high_freq / nyquist
            b, a = butter(4, [low_norm, high_norm], btype='band')
        elif filter_type == 'notch':
            notch_freq = self.config['notch_filter']
            b, a = iirnotch(notch_freq, 30.0, sampling_rate)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
        w, h = signal.freqz(b, a)
        freq = w * sampling_rate / (2 * np.pi)
        
        plt.figure(figsize=(10, 6))
        plt.plot(freq, 20 * np.log10(abs(h)))
        plt.title(f'Frequency Response - {filter_type} filter')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.show() 