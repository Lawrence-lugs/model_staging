#!/usr/bin/env python3
"""
Simplified version of DS-CNN quantization script for testing.
This version creates dummy data instead of downloading from HuggingFace.
"""

import os
import numpy as np
from torch.utils.data import Dataset
from quantization_scripts.quantize import quantize_onnx_model

class DummySpeechDataset(Dataset):
    """
    Dummy dataset that generates synthetic MFCC-like features for testing quantization.
    This mimics the expected input format for the DS-CNN model.
    """
    
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        # Generate random MFCC-like features: (1, 49, 10)
        # 1 channel, 49 time frames, 10 MFCC coefficients
        self.data = []
        
        for i in range(num_samples):
            # Generate synthetic MFCC features
            mfcc_features = np.random.randn(1, 49, 10).astype(np.float32)
            # Normalize to reasonable range
            mfcc_features = (mfcc_features - mfcc_features.mean()) / (mfcc_features.std() + 1e-8)
            
            # Random label (0-11 for 12 classes)
            label = np.random.randint(0, 12)
            
            self.data.append({
                'img': mfcc_features,  # Neural Compressor expects 'img' key
                'label': label
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def main():
    """
    Test quantization with dummy data.
    """
    print("=== DS-CNN Quantization Test (Dummy Data) ===")
    
    # Paths
    input_model_path = "onnx_models/ks_raw.onnx"
    output_model_path = "onnx_models/ks_quantized_int8_test.onnx"
    
    # Check if input model exists
    if not os.path.exists(input_model_path):
        print(f"Error: Input model {input_model_path} not found!")
        return
    
    print(f"Input model: {input_model_path}")
    print(f"Output model: {output_model_path}")
    
    # Create dummy dataset
    print("Creating dummy validation dataset...")
    dummy_dataset = DummySpeechDataset(num_samples=50)  # Small dataset for quick testing
    
    # Quantization parameters
    quantization_params = {
        "approach": "static",
        "batch_size": 1,
        "quant_level": 1,
        "quant_format": "QOperator"  # As requested
    }
    
    print(f"\nQuantization parameters:")
    for key, value in quantization_params.items():
        print(f"  {key}: {value}")
    
    # Perform quantization
    print(f"\nStarting quantization with {len(dummy_dataset)} dummy samples...")
    
    try:
        quantized_model = quantize_onnx_model(
            modelpath=input_model_path,
            test_set=dummy_dataset,
            **quantization_params
        )
        
        # Save quantized model
        quantized_model.save(output_model_path)
        print(f"\nQuantization completed successfully!")
        print(f"Quantized model saved to: {output_model_path}")
        
        # Print model file sizes for comparison
        original_size = os.path.getsize(input_model_path) / (1024 * 1024)  # MB
        quantized_size = os.path.getsize(output_model_path) / (1024 * 1024)  # MB
        compression_ratio = original_size / quantized_size
        
        print(f"\nModel size comparison:")
        print(f"  Original: {original_size:.2f} MB")
        print(f"  Quantized: {quantized_size:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
