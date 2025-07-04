#!/usr/bin/env python3
"""
Test script for FC_AE quantization using mock ToyADMOS data.
This version doesn't require external dataset downloads.
"""

import numpy as np
import os
from neural_compressor import quantization, PostTrainingQuantConfig
from neural_compressor.config import AccuracyCriterion

class MockToyADMOSDataLoader:
    """Mock DataLoader for testing FC_AE quantization without external dependencies"""
    
    def __init__(self, num_samples=100, batch_size=1):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.index = 0
        
        # Generate synthetic mel spectrogram data (217 time frames × 640 features for FC_AE)
        print(f"Generating {num_samples} synthetic ToyADMOS-like mel spectrograms [217, 640]...")
        self.data = self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate realistic synthetic mel spectrogram data with shape [217, 640]"""
        synthetic_data = []
        
        for i in range(self.num_samples):
            # Create mel spectrogram-like data with machine sound characteristics
            # Shape: [217, 640] following toycar_dset.py format
            mel_spec = np.random.randn(217, 640).astype(np.float32)
            
            # Add temporal patterns across the 217 time frames
            for t in range(217):
                # Add periodic patterns typical of machine sounds
                freq_pattern = np.sin(np.linspace(0, 6*np.pi, 640)) * 0.4
                amplitude_mod = np.cos(np.linspace(0, 2*np.pi, 640)) * 0.2
                
                # Apply temporal variation
                temporal_factor = np.sin(2 * np.pi * t / 217)
                mel_spec[t] += (freq_pattern + amplitude_mod) * temporal_factor
                
                # Add some noise bursts (anomaly patterns)
                if i % 10 == 0 and t % 20 == 0:  # Sparse anomalies
                    noise_start = np.random.randint(0, 500)
                    noise_end = min(noise_start + 140, 640)
                    mel_spec[t, noise_start:noise_end] += np.random.randn(noise_end - noise_start) * 0.8
            
            # Normalize following toycar_dset.py normalization
            std = 12.0590
            mean = -28.0861
            mel_spec = (mel_spec - mean) / std
            
            synthetic_data.append(mel_spec)
        
        return np.array(synthetic_data)
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= self.num_samples:
            raise StopIteration
        
        batch_data = []
        for _ in range(min(self.batch_size, self.num_samples - self.index)):
            batch_data.append(self.data[self.index])
            self.index += 1
        
        batch_array = np.array(batch_data)
        # For autoencoder: input == target
        return batch_array, batch_array
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

def test_quantize_fc_ae():
    """Test FC_AE quantization with mock data"""
    
    model_path = "onnx_models/ad_raw.onnx"
    output_path = "onnx_models/ad_quantized_int8_test.onnx"
    
    print("=" * 60)
    print("FC_AE ToyADMOS Test Quantization (Mock Data)")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please ensure the ONNX model exists before running the test.")
        return False
    
    # Create mock dataloader
    print("Creating mock ToyADMOS validation data...")
    dataloader = MockToyADMOSDataLoader(num_samples=50, batch_size=1)
    
    print(f"Generated {len(dataloader.data)} synthetic mel spectrograms")
    print(f"Input shape: {dataloader.data[0].shape}")
    print(f"Data range: [{dataloader.data.min():.3f}, {dataloader.data.max():.3f}]")
    
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
    success = test_quantize_fc_ae()
    if success:
        print("\n✅ FC_AE quantization test passed!")
    else:
        print("\n❌ FC_AE quantization test failed!")
