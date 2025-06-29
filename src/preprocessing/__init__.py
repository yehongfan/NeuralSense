"""
EEG Preprocessing Module

This module provides comprehensive preprocessing functionality for EEG data
used in Parkinson's disease detection.
"""

from .eeg_preprocessor import EEGPreprocessor
from .artifact_removal import ArtifactRemover
from .signal_filtering import SignalFilter

__all__ = ['EEGPreprocessor', 'ArtifactRemover', 'SignalFilter'] 