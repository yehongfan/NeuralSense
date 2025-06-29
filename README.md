<<<<<<< HEAD
# Parkinson's Disease Detection using EEG Data

This project implements machine learning models to detect Parkinson's disease using electroencephalogram (EEG) data.

## Project Overview

Parkinson's disease is a neurodegenerative disorder that affects movement and can be detected through changes in brain activity patterns captured by EEG. This project provides a complete pipeline for:

- EEG data preprocessing and feature extraction
- Machine learning model training and evaluation
- Real-time prediction capabilities
- Comprehensive analysis and visualization

## Features

- **Data Preprocessing**: Signal filtering, artifact removal, and feature extraction
- **Feature Engineering**: Time-domain, frequency-domain, and time-frequency features
- **Multiple ML Models**: Support Vector Machine, Random Forest, Neural Networks
- **Cross-validation**: Robust model evaluation
- **Visualization**: EEG signal plots, feature importance, and performance metrics
- **Real-time Processing**: Live EEG data processing capabilities

## Project Structure

```
eeg_research/
├── data/                   # Data storage
├── src/                    # Source code
│   ├── preprocessing/      # Data preprocessing modules
│   ├── features/          # Feature extraction
│   ├── models/            # ML models
│   ├── evaluation/        # Model evaluation
│   └── visualization/     # Plotting and visualization
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── config/               # Configuration files
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Preparation**: Place your EEG data in the `data/` directory
2. **Preprocessing**: Run the preprocessing pipeline
3. **Training**: Train the models using the provided scripts
4. **Evaluation**: Evaluate model performance
5. **Prediction**: Use trained models for new predictions

## Data Format

The system expects EEG data in the following format:
- Raw EEG signals (time series data)
- Sampling rate: 250-1000 Hz
- Channels: Standard 10-20 electrode placement
- Labels: Binary (0: Healthy, 1: Parkinson's)

## Model Performance

Current best model achieves:
- Accuracy: ~85-90%
- Sensitivity: ~88%
- Specificity: ~87%

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- EEG-based Parkinson's disease detection studies
- Machine learning in neuroimaging
- Signal processing for biomedical applications 
=======
# NeuralSense
>>>>>>> a5969b0c268103332f472315fd3354b46b07d487
