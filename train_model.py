#!/usr/bin/env python3
"""
Main Training Script for Parkinson's Disease Detection

This script demonstrates the complete pipeline for training machine learning models
to detect Parkinson's disease using EEG data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing import EEGPreprocessor
from features import FeatureExtractor
from models import ModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_sample_data(n_subjects: int = 100, n_channels: int = 19, 
                        sampling_rate: int = 250, duration: int = 60) -> tuple:
    """
    Generate sample EEG data for demonstration purposes.
    
    Args:
        n_subjects: Number of subjects
        n_channels: Number of EEG channels
        sampling_rate: Sampling rate in Hz
        duration: Recording duration in seconds
        
    Returns:
        Tuple of (eeg_data_list, labels)
    """
    logger.info("Generating sample EEG data...")
    
    eeg_data_list = []
    labels = []
    
    n_samples = sampling_rate * duration
    
    for i in range(n_subjects):
        # Generate synthetic EEG data
        # Healthy subjects have more alpha activity (8-13 Hz)
        # Parkinson's subjects have more beta activity (13-30 Hz) and reduced alpha
        
        if i < n_subjects // 2:  # Healthy subjects
            label = 0
            # Generate alpha-dominant activity
            alpha_freq = 10  # Hz
            alpha_phase = np.random.uniform(0, 2*np.pi, n_channels)
            alpha_amplitude = np.random.uniform(5, 15, n_channels)
            
            # Generate beta activity (lower in healthy subjects)
            beta_freq = 20  # Hz
            beta_phase = np.random.uniform(0, 2*np.pi, n_channels)
            beta_amplitude = np.random.uniform(1, 5, n_channels)
            
        else:  # Parkinson's subjects
            label = 1
            # Generate beta-dominant activity
            alpha_freq = 10  # Hz
            alpha_phase = np.random.uniform(0, 2*np.pi, n_channels)
            alpha_amplitude = np.random.uniform(1, 5, n_channels)  # Reduced alpha
            
            # Generate beta activity (higher in Parkinson's subjects)
            beta_freq = 20  # Hz
            beta_phase = np.random.uniform(0, 2*np.pi, n_channels)
            beta_amplitude = np.random.uniform(5, 15, n_channels)  # Increased beta
        
        # Generate time vector
        t = np.linspace(0, duration, n_samples)
        
        # Generate EEG data for each channel
        eeg_data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Alpha activity
            alpha_signal = alpha_amplitude[ch] * np.sin(2 * np.pi * alpha_freq * t + alpha_phase[ch])
            
            # Beta activity
            beta_signal = beta_amplitude[ch] * np.sin(2 * np.pi * beta_freq * t + beta_phase[ch])
            
            # Add noise
            noise = np.random.normal(0, 2, n_samples)
            
            # Combine signals
            eeg_data[ch, :] = alpha_signal + beta_signal + noise
        
        eeg_data_list.append(eeg_data)
        labels.append(label)
    
    logger.info(f"Generated data for {n_subjects} subjects: {sum(labels)} Parkinson's, {n_subjects - sum(labels)} healthy")
    
    return eeg_data_list, np.array(labels)

def main():
    """Main training pipeline."""
    logger.info("Starting Parkinson's Disease Detection Training Pipeline")
    
    # Create output directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Step 1: Generate sample data
    logger.info("Step 1: Generating sample EEG data...")
    eeg_data_list, labels = generate_sample_data(n_subjects=200)
    
    # Step 2: Preprocess EEG data
    logger.info("Step 2: Preprocessing EEG data...")
    preprocessor = EEGPreprocessor()
    
    all_segments = []
    all_segment_labels = []
    
    for i, eeg_data in enumerate(eeg_data_list):
        # Preprocess each subject's data
        segments = preprocessor.preprocess_raw_data(eeg_data, sampling_rate=250)
        all_segments.extend(segments)
        all_segment_labels.extend([labels[i]] * len(segments))
    
    all_segment_labels = np.array(all_segment_labels)
    
    logger.info(f"Preprocessed {len(all_segments)} segments from {len(eeg_data_list)} subjects")
    
    # Step 3: Extract features
    logger.info("Step 3: Extracting features...")
    feature_extractor = FeatureExtractor()
    
    # Extract features from all segments
    features_df = feature_extractor.fit_transform(all_segments, sampling_rate=250)
    
    logger.info(f"Extracted {features_df.shape[1]} features from {features_df.shape[0]} segments")
    
    # Step 4: Train models
    logger.info("Step 4: Training machine learning models...")
    model_trainer = ModelTrainer()
    
    # Train all models
    results = model_trainer.train_models(features_df, all_segment_labels)
    
    # Step 5: Display results
    logger.info("Step 5: Displaying results...")
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Step 6: Get best model
    best_model_name, best_model = model_trainer.get_best_model()
    print(f"\nBest Model: {best_model_name}")
    
    # Step 7: Feature importance (if available)
    try:
        feature_importance = model_trainer.get_feature_importance()
        print(f"\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
            print(f"  {i+1:2d}. {feature}: {importance:.4f}")
    except Exception as e:
        logger.warning(f"Could not get feature importance: {e}")
    
    # Step 8: Save models
    logger.info("Step 6: Saving models...")
    model_trainer.save_models()
    
    # Step 9: Create visualizations
    logger.info("Step 7: Creating visualizations...")
    
    # Plot feature importance
    try:
        feature_importance = model_trainer.get_feature_importance()
        top_features = dict(list(feature_importance.items())[:20])
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), list(top_features.values()))
        plt.yticks(range(len(top_features)), list(top_features.keys()))
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        logger.warning(f"Could not create feature importance plot: {e}")
    
    # Plot model comparison
    model_names = list(results.keys())
    accuracies = [results[name]['metrics']['accuracy'] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 10: Save results summary
    logger.info("Step 8: Saving results summary...")
    
    results_summary = {
        'n_subjects': len(eeg_data_list),
        'n_segments': len(all_segments),
        'n_features': features_df.shape[1],
        'class_distribution': {
            'healthy': int(sum(all_segment_labels == 0)),
            'parkinsons': int(sum(all_segment_labels == 1))
        },
        'model_performance': {
            model_name: {
                'accuracy': result['metrics']['accuracy'],
                'precision': result['metrics']['precision'],
                'recall': result['metrics']['recall'],
                'f1': result['metrics']['f1']
            } for model_name, result in results.items()
        },
        'best_model': best_model_name
    }
    
    import json
    with open('results/training_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Results saved to: results/training_summary.json")
    logger.info(f"Models saved to: models/")
    logger.info(f"Plots saved to: plots/")

if __name__ == "__main__":
    main() 