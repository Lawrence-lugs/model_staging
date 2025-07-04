#!/usr/bin/env python3
"""
Script to quantize DS-CNN keyword spotting model using Neural Compressor
with Google Speech Commands 2 dataset from HuggingFace.

This script:
1. Downloads the Google Speech Commands 2 validation set from HuggingFace
2. Preprocesses the audio data for DS-CNN model (MFCC features)
3. Quantizes the ks_raw.onnx model to 8-bit using QOperator format
"""

import os
import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
import librosa
from quantization_scripts.quantize import quantize_onnx_model

# MLPerfTiny keyword spotting 11 classes + silence + unknown
KS_CLASSES = [
    'silence', 'unknown', 'yes', 'no', 'up', 'down', 
    'left', 'right', 'on', 'off', 'stop', 'go'
]

class GoogleSpeechCommandsDataset(Dataset):
    """
    Dataset class for Google Speech Commands 2 data preprocessed for DS-CNN model.
    
    The DS-CNN model expects MFCC features as input:
    - Input shape: (1, 49, 10) - 1 channel, 49 time frames, 10 MFCC coefficients
    - Sampling rate: 16kHz
    - Window length: 30ms
    - Window step: 20ms
    """
    
    def __init__(self, hf_dataset, target_sample_rate=16000):
        self.dataset = hf_dataset
        self.target_sample_rate = target_sample_rate
        
        # Filter to only include the 11 keywords used in MLPerfTiny
        # Map HuggingFace labels to our KS_CLASSES
        self.label_mapping = self._create_label_mapping()
        self.filtered_data = self._filter_dataset()
        
    def _create_label_mapping(self):
        """Create mapping from HuggingFace speech commands to MLPerfTiny classes"""
        # HuggingFace speech commands has 35 classes, we only want 11 + silence + unknown
        target_keywords = {'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'}
        
        label_map = {}
        for i, item in enumerate(self.dataset):
            label_text = item['label']  # This should be the text label
            if label_text in target_keywords:
                if label_text not in label_map:
                    label_map[label_text] = KS_CLASSES.index(label_text)
            else:
                # Map everything else to 'unknown'
                if 'unknown' not in label_map:
                    label_map['unknown'] = KS_CLASSES.index('unknown')
                if label_text not in label_map:
                    label_map[label_text] = KS_CLASSES.index('unknown')
        
        return label_map
    
    def _filter_dataset(self):
        """Filter dataset to only include samples we want to use for quantization"""
        filtered = []
        max_samples_per_class = 50  # Limit samples for faster quantization
        class_counts = {}
        
        for item in self.dataset:
            label_text = item['label']
            mapped_label = self.label_mapping.get(label_text, self.label_mapping['unknown'])
            
            # Count samples per class to ensure balanced dataset
            if mapped_label not in class_counts:
                class_counts[mapped_label] = 0
                
            if class_counts[mapped_label] < max_samples_per_class:
                filtered.append({
                    'audio': item['audio'], 
                    'label': mapped_label,
                    'original_label': label_text
                })
                class_counts[mapped_label] += 1
        
        print(f"Filtered dataset: {len(filtered)} samples")
        print(f"Class distribution: {class_counts}")
        return filtered
    
    def _extract_mfcc_features(self, audio_array, sample_rate):
        """
        Extract MFCC features matching DS-CNN input requirements.
        
        Returns:
            numpy array of shape (49, 10) - 49 time frames, 10 MFCC coefficients
        """
        # Ensure audio is 1 second long (pad or trim)
        target_length = self.target_sample_rate  # 16000 samples for 1 second
        
        if len(audio_array) > target_length:
            # Trim to 1 second
            audio_array = audio_array[:target_length]
        elif len(audio_array) < target_length:
            # Pad with zeros
            audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
        
        # Extract MFCC features
        # Using librosa for consistency with common ML practices
        mfccs = librosa.feature.mfcc(
            y=audio_array.astype(np.float32),
            sr=sample_rate,
            n_mfcc=10,        # 10 MFCC coefficients
            n_fft=512,        # FFT window size
            hop_length=320,   # ~20ms hop (320/16000)
            win_length=480,   # ~30ms window (480/16000)
            n_mels=40,        # Number of mel filters
            fmax=8000         # Maximum frequency
        )
        
        # Transpose to get (time, features) shape and ensure we have 49 frames
        mfccs = mfccs.T  # Shape: (time_frames, 10)
        
        # Pad or trim to exactly 49 frames
        if mfccs.shape[0] > 49:
            mfccs = mfccs[:49, :]
        elif mfccs.shape[0] < 49:
            mfccs = np.pad(mfccs, ((0, 49 - mfccs.shape[0]), (0, 0)))
        
        return mfccs
    
    def __len__(self):
        return len(self.filtered_data)
    
    def __getitem__(self, idx):
        item = self.filtered_data[idx]
        audio_data = item['audio']
        
        # Get audio array and sample rate
        audio_array = audio_data['array']
        sample_rate = audio_data['sampling_rate']
        
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            # Use torchaudio for resampling
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)
            audio_array = audio_tensor.squeeze(0).numpy()
            sample_rate = self.target_sample_rate
        
        # Extract MFCC features
        mfcc_features = self._extract_mfcc_features(audio_array, sample_rate)
        
        # Add channel dimension: (1, 49, 10)
        mfcc_features = np.expand_dims(mfcc_features, axis=0)
        
        return {
            'img': mfcc_features.astype(np.float32),  # Neural Compressor expects 'img' key
            'label': item['label']
        }


def load_speech_commands_validation_set():
    """
    Load Google Speech Commands 2 validation set from HuggingFace.
    
    Returns:
        GoogleSpeechCommandsDataset: Preprocessed dataset ready for quantization
    """
    print("Loading Google Speech Commands 2 validation set from HuggingFace...")

    import aiohttp
    
    # Load only the validation split to save time and bandwidth
    dataset = load_dataset(
        "speech_commands", 
        "v0.02",
        split="validation",
        trust_remote_code=True
    )
    
    print(f"Loaded {len(dataset)} validation samples")
    
    # Create our custom dataset with preprocessing
    ks_dataset = GoogleSpeechCommandsDataset(dataset)
    
    return ks_dataset


def main():
    """
    Main function to quantize the DS-CNN keyword spotting model.
    """
    print("=== DS-CNN Keyword Spotting Model Quantization ===")
    
    # Paths
    input_model_path = "onnx_models/ks_raw.onnx"
    output_model_path = "onnx_models/ks_quantized_int8.onnx"
    
    # Check if input model exists
    if not os.path.exists(input_model_path):
        print(f"Error: Input model {input_model_path} not found!")
        return
    
    print(f"Input model: {input_model_path}")
    print(f"Output model: {output_model_path}")
    
    # Load validation dataset
    try:
        validation_dataset = load_speech_commands_validation_set()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have 'datasets' and 'librosa' installed:")
        print("pip install datasets librosa")
        return
    
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
    print(f"\nStarting quantization...")
    print(f"Using {len(validation_dataset)} samples for calibration")
    
    try:
        quantized_model = quantize_onnx_model(
            modelpath=input_model_path,
            test_set=validation_dataset,
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
