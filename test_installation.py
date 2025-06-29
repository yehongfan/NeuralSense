#!/usr/bin/env python3
"""
Test Installation Script

This script tests that all dependencies are properly installed and the modules
can be imported correctly.
"""

import sys
import importlib
from pathlib import Path

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        if package_name:
            module = importlib.import_module(module_name, package=package_name)
        else:
            module = importlib.import_module(module_name)
        print(f"✓ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {module_name} import failed: {e}")
        return False

def main():
    """Test all required dependencies and modules."""
    print("Testing Parkinson's Disease EEG Detection Installation")
    print("=" * 60)
    
    # Add src to path
    src_path = Path(__file__).parent / "src"
    sys.path.append(str(src_path))
    
    # Test core scientific libraries
    print("\n1. Testing Core Scientific Libraries:")
    core_libraries = [
        "numpy",
        "scipy", 
        "pandas",
        "matplotlib",
        "seaborn"
    ]
    
    core_success = True
    for lib in core_libraries:
        if not test_import(lib):
            core_success = False
    
    # Test machine learning libraries
    print("\n2. Testing Machine Learning Libraries:")
    ml_libraries = [
        "sklearn",
        "tensorflow",
        "keras"
    ]
    
    ml_success = True
    for lib in ml_libraries:
        if not test_import(lib):
            ml_success = False
    
    # Test EEG processing libraries
    print("\n3. Testing EEG Processing Libraries:")
    eeg_libraries = [
        "mne",
        "pywt",
        "neurokit2"
    ]
    
    eeg_success = True
    for lib in eeg_libraries:
        if not test_import(lib):
            eeg_success = False
    
    # Test our custom modules
    print("\n4. Testing Custom Modules:")
    custom_modules = [
        "preprocessing",
        "features", 
        "models"
    ]
    
    custom_success = True
    for module in custom_modules:
        if not test_import(module):
            custom_success = False
    
    # Test specific classes
    print("\n5. Testing Specific Classes:")
    try:
        from preprocessing import EEGPreprocessor
        print("✓ EEGPreprocessor imported successfully")
        
        from features import FeatureExtractor
        print("✓ FeatureExtractor imported successfully")
        
        from models import ModelTrainer
        print("✓ ModelTrainer imported successfully")
        
        class_success = True
    except ImportError as e:
        print(f"✗ Class import failed: {e}")
        class_success = False
    
    # Summary
    print("\n" + "=" * 60)
    print("INSTALLATION TEST SUMMARY")
    print("=" * 60)
    
    if all([core_success, ml_success, eeg_success, custom_success, class_success]):
        print("✓ All tests passed! Installation is complete and ready to use.")
        print("\nYou can now run:")
        print("  python train_model.py")
        print("  jupyter notebook notebooks/parkinsons_eeg_analysis.ipynb")
    else:
        print("✗ Some tests failed. Please install missing dependencies:")
        print("\nMissing core libraries:")
        if not core_success:
            print("  pip install numpy scipy pandas matplotlib seaborn")
        
        print("\nMissing ML libraries:")
        if not ml_success:
            print("  pip install scikit-learn tensorflow keras")
        
        print("\nMissing EEG libraries:")
        if not eeg_success:
            print("  pip install mne pywt neurokit2")
        
        print("\nMissing custom modules:")
        if not custom_success:
            print("  Check that the src/ directory structure is correct")
    
    print("\nFor complete installation, run:")
    print("  pip install -r requirements.txt")

if __name__ == "__main__":
    main() 