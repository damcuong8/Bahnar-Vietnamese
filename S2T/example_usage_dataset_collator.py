"""
Example usage of ViBaSpeechToTextDataset and DataCollatorSpeechToText

This script demonstrates how to use the custom dataset and data collator
for training SeamlessM4T model with Vietnamese and English text.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from train_kaggle import ViBaSpeechToTextDataset, DataCollatorSpeechToText
from seamless_feature_extractor import SeamlessM4TFeatureExtractor


def main():
    # Initialize processor and feature extractor
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    feature_extractor = SeamlessM4TFeatureExtractor(
        feature_size=80,
        sampling_rate=16000,
        num_mel_bins=80,
        padding_value=0.0,
        stride=2,
    )
    
    # Create dataset
    # Your Excel file should have columns: "source", "Tiếng Việt", "Tiếng Anh"
    # "source" column contains audio file paths or URLs
    dataset = ViBaSpeechToTextDataset(
        excel_path="path/to/your/data.xlsx",
        audio_col="source",
        vi_col="Tiếng Việt",
        en_col="Tiếng Anh",
        target_sr=16000,
        mono=True,
        augment_fn=None,  # Add your audio augmentation function here if needed
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create data collator
    data_collator = DataCollatorSpeechToText(
        feature_extractor=feature_extractor,
        processor=processor,
        padding=True,  # Dynamic padding
        pad_to_multiple_of=2,  # For efficiency
        target_language="vi",  # Vietnamese as target/labels
        pivot_language="en",   # English as pivot/context
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,  # Use multiple workers for parallel data loading
        pin_memory=True,  # For faster GPU transfer
    )
    
    # Iterate through batches
    print("\n=== Testing DataLoader ===")
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  audio_input_features shape: {batch['audio_input_features'].shape}")
        print(f"  audio_attention_mask shape: {batch['audio_attention_mask'].shape}")
        
        if "labels" in batch:
            print(f"  labels shape: {batch['labels'].shape}")
            print(f"  Number of valid tokens (not -100): {(batch['labels'] != -100).sum().item()}")
        
        if "text_input_pivot_ids" in batch:
            print(f"  text_input_pivot_ids shape: {batch['text_input_pivot_ids'].shape}")
            print(f"  text_pivot_attention_mask shape: {batch['text_pivot_attention_mask'].shape}")
        
        # Only test first batch
        if batch_idx == 0:
            break
    
    print("\n=== Testing Single Sample ===")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"  waveform shape: {sample['waveform'].shape}")
    print(f"  raw_vi: {sample['raw_vi'][:100]}...")  # First 100 chars
    print(f"  raw_en: {sample['raw_en'][:100] if sample['raw_en'] else 'None'}...")
    if sample['vi_input_ids'] is not None:
        print(f"  vi_input_ids shape: {sample['vi_input_ids'].shape}")
    if sample['en_input_ids'] is not None:
        print(f"  en_input_ids shape: {sample['en_input_ids'].shape}")
    print(f"  source: {sample['source']}")
    
    print("\n✅ Dataset and DataCollator working correctly!")


if __name__ == "__main__":
    main()


