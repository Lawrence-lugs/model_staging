#!/usr/bin/env python3
"""
Test script for ResNet CIFAR-10 quantization using single sample approach.
This version uses one sample repeated multiple times to avoid rank errors.
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from neural_compressor import quantization, PostTrainingQuantConfig
from neural_compressor.config import AccuracyCriterion

class SingleCIFAR10DataLoader:
    """DataLoader for testing ResNet quantization with single sample"""
    
    def __init__(self, num_iterations=50, batch_size=1):
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.current_iteration = 0
        
        # Generate a single CIFAR-10-like image (3, 32, 32)
        print("Generating synthetic CIFAR-10 image for testing...")
        self.sample_image = self._generate_sample()
        
    def _generate_sample(self):
        """Generate a single realistic CIFAR-10-like image"""
        # Create RGB image (3, 32, 32)
        image = np.random.randn(3, 32, 32).astype(np.float32)
        
        # Add spatial patterns to make it more realistic
        for c in range(3):
            for x in range(32):
                for y in range(32):
                    # Add some structure
                    spatial_pattern = (
                        np.sin(x / 4) * np.cos(y / 4) +
                        np.sin(x / 8) * np.sin(y / 8) * 0.5
                    ) * 0.3
                    image[c, x, y] += spatial_pattern
        
        # Normalize to typical ImageNet-normalized range
        # Mean ≈ 0, std ≈ 1 after normalization
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        # Clip to reasonable range
        image = np.clip(image, -3, 3)
        
        print(f"Generated sample shape: {image.shape}")
        print(f"Sample statistics: mean={image.mean():.3f}, std={image.std():.3f}")
        print(f"Sample range: [{image.min():.3f}, {image.max():.3f}]")
        
        return image
    
    def __iter__(self):
        self.current_iteration = 0
        return self
    
    def __next__(self):
        if self.current_iteration >= self.num_iterations:
            raise StopIteration
        
        self.current_iteration += 1
        # Return the same sample each time
        return self.sample_image, 0  # Label doesn't matter for calibration
    
    def __len__(self):
        return self.num_iterations

def test_quantize_resnet():
    """Test ResNet quantization with single sample"""
    
    model_path = "onnx_models/ic_raw.onnx"
    output_path = "onnx_models/ic_quantized_int8_test.onnx"
    
    print("=" * 60)
    print("ResNet CIFAR-10 Test Quantization (Single Sample)")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please ensure the ONNX model exists before running the test.")
        return False
    
    # Create test dataloader
    print("Creating test CIFAR-10 data...")
    dataloader = SingleCIFAR10DataLoader(num_iterations=50, batch_size=1)
    
    # Test the dataloader
    test_iter = iter(dataloader)
    test_image, test_label = next(test_iter)
    print(f"Test image shape: {test_image.shape}")
    print(f"Test label: {test_label}")
    
    # Set up quantization
    accuracy_criterion = AccuracyCriterion()
    accuracy_criterion.relative = 0.02
    
    config = PostTrainingQuantConfig(
        approach="static",
        quant_level=1,
        quant_format="QOperator"
    )
    
    print(f"\nTesting quantization...")
    print(f"Approach: static")
    print(f"Quant Level: 1 (8-bit)")
    print(f"Format: QOperator")
    print(f"Input shape: {test_image.shape}")
    
    try:
        # Perform quantization
        quantized_model = quantization.fit(
            model=model_path,
            conf=config,
            calib_dataloader=dataloader,
            accuracy_criterion=accuracy_criterion
        )
        
        # Save quantized model
        quantized_model.save(output_path)
        
        # Report results
        original_size = os.path.getsize(model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print("\n" + "=" * 60)
        print("TEST QUANTIZATION COMPLETED!")
        print("=" * 60)
        print(f"Original: {original_size:.2f} MB")
        print(f"Quantized: {quantized_size:.2f} MB")
        print(f"Compression: {original_size/quantized_size:.2f}x")
        print(f"Test output: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quantize_resnet()
    if success:
        print("\n✅ ResNet quantization test passed!")
        print("🖼️ Successfully quantized with CIFAR-10 format (3, 32, 32)")
    else:
        print("\n❌ ResNet quantization test failed!")
