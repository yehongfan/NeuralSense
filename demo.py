#!/usr/bin/env python3
"""
Demo Script for Parkinson's Disease Detection

This script demonstrates the basic usage of the Parkinson's disease detection
pipeline with a small sample dataset.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from preprocessing import EEGPreprocessor
from features import FeatureExtractor
from models import ModelTrainer

def generate_mini_dataset():
    """Generate a small dataset for demonstration."""
    print("Generating mini dataset...")
    
    # Generate 20 subjects (10 healthy, 10 Parkinson's)
    n_subjects = 20
    n_channels = 19
    sampling_rate = 250
    duration = 30  # 30 seconds each
    
    eeg_data_list = []
    labels = []
    
    n_samples = sampling_rate * duration
    
    for i in range(n_subjects):
        # Generate synthetic EEG data
        if i < n_subjects // 2:  # Healthy subjects
            label = 0
            alpha_amplitude = np.random.uniform(8, 12, n_channels)
            beta_amplitude = np.random.uniform(2, 4, n_channels)
        else:  # Parkinson's subjects
            label = 1
            alpha_amplitude = np.random.uniform(2, 4, n_channels)  # Reduced alpha
            beta_amplitude = np.random.uniform(8, 12, n_channels)  # Increased beta
        
        # Generate time vector
        t = np.linspace(0, duration, n_samples)
        
        # Generate EEG data for each channel
        eeg_data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Alpha activity (10 Hz)
            alpha_signal = alpha_amplitude[ch] * np.sin(2 * np.pi * 10 * t)
            
            # Beta activity (20 Hz)
            beta_signal = beta_amplitude[ch] * np.sin(2 * np.pi * 20 * t)
            
            # Add noise
            noise = np.random.normal(0, 1, n_samples)
            
            # Combine signals
            eeg_data[ch, :] = alpha_signal + beta_signal + noise
        
        eeg_data_list.append(eeg_data)
        labels.append(label)
    
    return eeg_data_list, np.array(labels)

def main():
    """Run the demo pipeline."""
    print("=" * 60)
    print("PARKINSON'S DISEASE DETECTION DEMO")
    print("=" * 60)
    
    # Step 1: Generate data
    eeg_data_list, labels = generate_mini_dataset()
    print(f"Generated data for {len(eeg_data_list)} subjects")
    print(f"  - Healthy: {sum(labels == 0)}")
    print(f"  - Parkinson's: {sum(labels == 1)}")
    
    # Step 2: Preprocess data
    print("\nStep 1: Preprocessing EEG data...")
    preprocessor = EEGPreprocessor()
    
    all_segments = []
    all_segment_labels = []
    
    for i, eeg_data in enumerate(eeg_data_list):
        segments = preprocessor.preprocess_raw_data(eeg_data, sampling_rate=250)
        all_segments.extend(segments)
        all_segment_labels.extend([labels[i]] * len(segments))
    
    all_segment_labels = np.array(all_segment_labels)
    print(f"Preprocessed {len(all_segments)} segments")
    
    # Step 3: Extract features
    print("\nStep 2: Extracting features...")
    feature_extractor = FeatureExtractor()
    
    # Use a simplified feature set for demo
    feature_extractor.config = {
        'time_domain': ['mean', 'std', 'variance', 'skewness', 'kurtosis'],
        'frequency_domain': ['spectral_entropy', 'spectral_centroid'],
        'time_frequency': [],
        'connectivity': []
    }
    
    features_df = feature_extractor.fit_transform(all_segments, sampling_rate=250)
    print(f"Extracted {features_df.shape[1]} features")
    
    # Step 4: Train models
    print("\nStep 3: Training models...")
    model_trainer = ModelTrainer()
    
    # Use simplified model config
    model_trainer.config = {
        'training': {
            'test_size': 0.3,
            'random_state': 42
        },
        'models': {
            'svm': {'kernel': 'rbf', 'C': 1.0},
            'random_forest': {'n_estimators': 50, 'max_depth': 5}
        }
    }
    
    results = model_trainer.train_models(features_df, all_segment_labels)
    
    # Step 5: Display results
    print("\nStep 4: Results")
    print("-" * 40)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1']:.3f}")
        print()
    
    # Get best model
    best_model_name, best_model = model_trainer.get_best_model()
    print(f"Best Model: {best_model_name}")
    print(f"Best Accuracy: {results[best_model_name]['metrics']['accuracy']:.3f}")
    
    # Step 6: Make a prediction
    print("\nStep 5: Making a prediction...")
    
    # Use the first segment as a test sample
    test_segment = all_segments[0]
    test_features = feature_extractor.extract_features(test_segment, sampling_rate=250)
    
    prediction = model_trainer.predict(test_features)
    true_label = all_segment_labels[0]
    
    print(f"Test sample prediction: {prediction[0]} (True: {true_label})")
    prediction_text = "Parkinson's" if prediction[0] == 1 else "Healthy"
    true_text = "Parkinson's" if true_label == 1 else "Healthy"
    print(f"Prediction: {prediction_text}")
    print(f"True label: {true_text}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThis demo shows the basic pipeline:")
    print("1. Generate/prepare EEG data")
    print("2. Preprocess signals (filtering, segmentation)")
    print("3. Extract features (time/frequency domain)")
    print("4. Train machine learning models")
    print("5. Make predictions on new data")
    print("\nFor a full analysis, run:")
    print("  python train_model.py")
    print("  jupyter notebook notebooks/parkinsons_eeg_analysis.ipynb")

if __name__ == "__main__":
    main() 