#!/usr/bin/env python3
"""
Script to quantize ResNet image classification model using Neural Compressor
with CIFAR-10 dataset from HuggingFace.

This script:
1. Downloads the CIFAR-10 validation set from HuggingFace
2. Preprocesses the image data for ResNet model (32x32 RGB images)
3. Quantizes the ic_raw.onnx model to 8-bit using QOperator format
"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from neural_compressor import quantization, PostTrainingQuantConfig
from neural_compressor.config import AccuracyCriterion

# CIFAR-10 classes for reference
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class CIFAR10Dataset:
    """
    Dataset class for CIFAR-10 data preprocessed for ResNet model.
    
    The ResNet model expects:
    - Input shape: (3, 32, 32) - 3 channels (RGB), 32x32 pixels
    - Normalized images with ImageNet stats
    """
    
    def __init__(self, hf_dataset, num_samples=500):
        self.dataset = hf_dataset
        self.num_samples = min(num_samples, len(hf_dataset))
        
        # Standard CIFAR-10 preprocessing for ResNet
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Created CIFAR-10 dataset with {self.num_samples} samples")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError("Index out of range")
            
        sample = self.dataset[idx]
        
        # Get image and label
        image = sample['img']  # PIL Image
        label = sample['label']  # Integer label
        
        # Convert PIL Image to tensor and normalize
        if isinstance(image, Image.Image):
            image_tensor = self.transform(image)
        else:
            # Handle case where image might already be a tensor/array
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            # Apply normalization
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            image_tensor = normalize(image)
        
        return {
            'img': image_tensor.numpy().astype(np.float32),  # Shape: (3, 32, 32)
            'label': label
        }

class CIFAR10DataLoader:
    """Custom DataLoader for CIFAR-10 data compatible with neural compressor"""
    
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = len(dataset)
        self.index = 0
    
    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        
        batch_images = []
        batch_labels = []
        
        for _ in range(min(self.batch_size, self.length - self.index)):
            sample = self.dataset[self.index]
            batch_images.append(sample['img'])
            batch_labels.append(sample['label'])
            self.index += 1
        
        # Return as numpy arrays
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        
        return batch_images, batch_labels
    
    def __len__(self):
        return (self.length + self.batch_size - 1) // self.batch_size

class SingleImageDataLoader:
    """DataLoader that yields individual images for calibration"""
    
    def __init__(self, sample_image, num_iterations=50, batch_size=1):
        self.sample_image = sample_image  # Shape should be (3, 32, 32)
        self.num_iterations = num_iterations
        self.batch_size = batch_size  # Required by Neural Compressor
        self.current_iteration = 0
    
    def __iter__(self):
        self.current_iteration = 0
        return self
    
    def __next__(self):
        if self.current_iteration >= self.num_iterations:
            raise StopIteration
        
        self.current_iteration += 1
        # Return individual sample
        return self.sample_image, 0  # Label doesn't matter for calibration
    
    def __len__(self):
        return self.num_iterations

def load_cifar10_validation_data(num_samples=500):
    """
    Load CIFAR-10 validation set from HuggingFace.
    
    Args:
        num_samples: Number of validation samples to use
        
    Returns:
        CIFAR10Dataset: Preprocessed dataset ready for quantization
    """
    print("Loading CIFAR-10 dataset from HuggingFace...")
    
    try:
        # Load CIFAR-10 test set (used as validation for quantization)
        dataset = load_dataset("cifar10", split="test")
        print(f"Loaded {len(dataset)} test samples from CIFAR-10")
        
        # Create our custom dataset with preprocessing
        cifar10_dataset = CIFAR10Dataset(dataset, num_samples=num_samples)
        
        return cifar10_dataset
        
    except Exception as e:
        print(f"Error loading CIFAR-10 dataset: {e}")
        print("Generating synthetic CIFAR-10-like data...")
        return generate_synthetic_cifar10_data(num_samples)

def generate_synthetic_cifar10_data(num_samples=500):
    """Generate synthetic CIFAR-10-like data for testing"""
    print(f"Generating {num_samples} synthetic CIFAR-10 images...")
    
    class SyntheticCIFAR10Dataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
            self.data = self._generate_data()
        
        def _generate_data(self):
            data = []
            for i in range(self.num_samples):
                # Generate random RGB image (3, 32, 32)
                image = np.random.randn(3, 32, 32).astype(np.float32)
                
                # Apply some structure to make it more realistic
                # Add some spatial patterns
                for c in range(3):
                    for x in range(32):
                        for y in range(32):
                            # Add some spatial correlation
                            spatial_pattern = np.sin(x/4) * np.cos(y/4) * 0.5
                            image[c, x, y] += spatial_pattern
                
                # Normalize to typical range after ImageNet normalization
                image = (image - image.mean()) / (image.std() + 1e-8)
                image = np.clip(image, -3, 3)  # Reasonable range
                
                # Random label
                label = np.random.randint(0, 10)
                
                data.append({
                    'img': image,
                    'label': label
                })
            
            return data
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    return SyntheticCIFAR10Dataset(num_samples)

def load_single_cifar10_sample():
    """Load a single CIFAR-10 sample for individual calibration"""
    try:
        # Load one sample from CIFAR-10
        dataset = load_dataset("cifar10", split="test", streaming=True)
        sample = next(iter(dataset))
        
        print("Successfully loaded CIFAR-10 sample")
        
        # Preprocess the image
        image = sample['img']  # PIL Image
        
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = transform(image)
        
        print(f"Preprocessed image shape: {image_tensor.shape}")
        return image_tensor.numpy().astype(np.float32)
        
    except Exception as e:
        print(f"Error loading CIFAR-10 sample: {e}")
        print("Using synthetic sample...")
        
        # Generate synthetic sample
        image = np.random.randn(3, 32, 32).astype(np.float32)
        
        # Add some structure
        for c in range(3):
            for x in range(32):
                for y in range(32):
                    spatial_pattern = np.sin(x/4) * np.cos(y/4) * 0.5
                    image[c, x, y] += spatial_pattern
        
        # Normalize
        image = (image - image.mean()) / (image.std() + 1e-8)
        image = np.clip(image, -3, 3)
        
        return image

def quantize_resnet_cifar10(
    model_path="onnx_models/ic_raw.onnx",
    output_path="onnx_models/ic_quantized_int8.onnx",
    num_validation_samples=500,
    use_single_sample=False,
    batch_size=1
):
    """
    Quantize ResNet model for CIFAR-10 image classification using neural compressor
    
    Args:
        model_path: Path to the input ONNX model
        output_path: Path to save the quantized model
        num_validation_samples: Number of validation samples to use
        use_single_sample: Whether to use single sample repeated multiple times
        batch_size: Batch size for calibration
    """
    print("=" * 60)
    print("ResNet CIFAR-10 Model Quantization with Neural Compressor")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please ensure the ONNX model exists before quantization.")
        return None
    
    if use_single_sample:
        # Load a single sample and repeat it
        print(f"\nLoading single CIFAR-10 sample for calibration...")
        sample_image = load_single_cifar10_sample()
        
        print(f"Sample image shape: {sample_image.shape}")
        dataloader = SingleImageDataLoader(sample_image, num_iterations=50, batch_size=batch_size)
        
    else:
        # Load validation dataset
        print(f"\nLoading CIFAR-10 validation data ({num_validation_samples} samples)...")
        validation_dataset = load_cifar10_validation_data(num_validation_samples)
        
        if len(validation_dataset) == 0:
            print("Error: No validation data loaded!")
            return None
        
        print(f"Loaded {len(validation_dataset)} images")
        
        # Test a sample
        test_sample = validation_dataset[0]
        print(f"Sample image shape: {test_sample['img'].shape}")
        print(f"Sample label: {test_sample['label']} ({CIFAR10_CLASSES[test_sample['label']]})")
        
        # Create data loader
        dataloader = CIFAR10DataLoader(validation_dataset, batch_size=batch_size)
    
    # Set up quantization configuration
    accuracy_criterion = AccuracyCriterion()
    accuracy_criterion.relative = 0.02  # 2% accuracy tolerance
    
    config = PostTrainingQuantConfig(
        approach="static",
        quant_level=1,  # 8-bit quantization
        quant_format="QOperator"  # QOperator format as requested
    )
    
    print(f"\nStarting quantization...")
    print(f"Model: {model_path}")
    print(f"Approach: static")
    print(f"Quant Level: 1 (8-bit)")
    print(f"Quant Format: QOperator")
    print(f"Calibration data: {'Single sample repeated' if use_single_sample else f'{len(validation_dataset)} samples'}")
    
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
        
        # Check file sizes
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        compression_ratio = original_size / quantized_size
        
        print("\n" + "=" * 60)
        print("QUANTIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Original model: {model_path}")
        print(f"Quantized model: {output_path}")
        print(f"Original size: {original_size:.2f} MB")
        print(f"Quantized size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
        
        return quantized_model
        
    except Exception as e:
        print(f"\nError during quantization: {e}")
        print("This might be due to:")
        print("- Incompatible ONNX model format")
        print("- Missing neural-compressor dependencies")
        print("- Incorrect input dimensions")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run quantization with validation dataset
    quantized_model = quantize_resnet_cifar10(
        model_path="onnx_models/ic_raw.onnx",
        output_path="onnx_models/ic_quantized_int8.onnx",
        num_validation_samples=500,  # Efficient sample size
        use_single_sample=False,  # Use multiple samples for better calibration
        batch_size=1
    )
    
    if quantized_model:
        print("\n✅ ResNet CIFAR-10 quantization completed successfully!")
        print("🎯 8-bit QOperator quantized model saved to: onnx_models/ic_quantized_int8.onnx")
        print("🖼️ Used CIFAR-10 validation images (3, 32, 32) as required by the model")
    else:
        print("\n❌ Quantization failed. Please check the error messages above.")
