"""
Dataset and data collator classes for SeamlessM4T v2 training
Extracted from train_kaggle.py for better modularity
"""

import random
import logging
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass

import torch
import torchaudio
from torchaudio.transforms import Resample
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoProcessor

from seamless_feature_extractor import SeamlessM4TFeatureExtractor


logger = logging.getLogger(__name__)


class DummySpeechToTextDataset(Dataset):
    """
    Placeholder dataset for demonstration purposes.
    Replace this with your actual dataset implementation.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        max_audio_length: int = 30,
        max_text_length: int = 200,
        sample_rate: int = 16000,
        num_features: int = 160,
    ):
        self.num_samples = num_samples
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.sample_rate = sample_rate
        self.num_features = num_features
        
        logger.info(
            f"Created dummy dataset with {num_samples} samples "
            f"(max_audio_length={max_audio_length}s, max_text_length={max_text_length} tokens)"
        )
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns a dummy sample.
        In production, replace this with actual data loading.
        
        Returns:
            dict with:
                - audio_input_features: Audio features (seq_len, num_features)
                - text_input_pivot_ids: Text input IDs for pivot (teacher) 
                - labels: Target token IDs
                - audio_attention_mask: Attention mask for audio
                - text_pivot_attention_mask: Attention mask for text pivot
        """
        # Random audio length (in frames, not seconds)
        audio_len = random.randint(100, self.max_audio_length * 50)  # ~50 frames per second
        
        # Random text length
        text_len = random.randint(10, self.max_text_length)
        
        # Generate dummy audio features
        audio_features = torch.randn(audio_len, self.num_features)
        audio_attention_mask = torch.ones(audio_len)
        
        # Generate dummy text input (pivot) and labels
        text_input_ids = torch.randint(4, 1000, (text_len,))  # Avoid special tokens 0-3
        text_attention_mask = torch.ones(text_len)
        
        # Labels (same as text for dummy data)
        labels = torch.randint(4, 1000, (text_len,))
        
        return {
            "audio_input_features": audio_features,
            "text_input_pivot_ids": text_input_ids,
            "labels": labels,
            "audio_attention_mask": audio_attention_mask,
            "text_pivot_attention_mask": text_attention_mask,
        }

    
class ViBaSpeechToTextDataset(Dataset):
    """
    Dataset for training SeamlessM4Tv2ForSpeechToTextTrain_Pivot model.
    
    Args:
        excel_path: Path to Excel file containing data
        audio_col: Column name containing audio file paths or URLs (default: "source")
        vi_col: Column name containing Vietnamese text (default: "Tiếng Việt")
        en_col: Column name containing English text (default: "Tiếng Anh")
        target_sr: Target sample rate for audio (default: 16000)
        mono: Convert audio to mono if True (default: True)
        augment_fn: Optional audio augmentation function
        use_cache: Enable in-memory caching of processed samples
    """

    def __init__(
        self,
        excel_path: str,
        audio_col: str = "source",
        vi_col: str = "Tiếng Việt",
        en_col: str = "Tiếng Anh",
        target_sr: int = 16000,
        mono: bool = True,
        augment_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        self.df = pd.read_excel(excel_path)
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.audio_col = audio_col
        self.vi_col = vi_col
        self.en_col = en_col
        self.target_sr = target_sr
        self.mono = mono
        self.augment_fn = augment_fn
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
        
        logger.info(f"Loaded dataset with {len(self.df)} samples from {excel_path}")
        if use_cache:
            logger.info("In-memory caching enabled")
    
    def _load_audio(self, audio_source: str) -> tuple:
        """
        Load audio from local file.
        Returns (waveform, sample_rate)
        """
        try:
            waveform, sr = torchaudio.load(audio_source)
            return waveform, sr
        except Exception as e:
            logger.error(f"Failed to load audio from {audio_source}: {e}")
            raise

    def _process_audio(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Process audio: convert to mono, resample, apply augmentation"""
        waveform = waveform.float()

        # Convert to mono if needed
        if self.mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            resampler = Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            sr = self.target_sr

        # Squeeze to 1D tensor
        waveform = waveform.squeeze(0)

        # Apply augmentation if provided
        if self.augment_fn is not None:
            waveform = self.augment_fn(waveform, sr)
        
        return waveform
    
    def _tokenize_text(self, text: str) -> tuple:
        """Tokenize text and return token IDs and attention mask"""
        if not text:
            return None, None
        
        encoded = self.processor(
            text=text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=4096,
        )
        tokens = encoded["input_ids"].squeeze(0)  # Shape: [seq_len]
        attention_mask = encoded["attention_mask"].squeeze(0)  # Shape: [seq_len]
        
        return tokens, attention_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Check cache first
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        
        row = self.df.iloc[idx]
        
        # Get audio source
        audio_source = str(row[self.audio_col]).strip()
        
        # Get text data
        vi_text = str(row[self.vi_col]) if not pd.isna(row[self.vi_col]) else ""
        en_text = str(row[self.en_col]) if (self.en_col in self.df.columns and not pd.isna(row[self.en_col])) else None

        # Load and process audio
        waveform, sr = self._load_audio(audio_source)
        waveform = self._process_audio(waveform, sr)

        # Tokenize texts
        vi_tokens, vi_attention_mask = self._tokenize_text(vi_text)
        en_tokens, en_attention_mask = self._tokenize_text(en_text)

        result = {
            "waveform": waveform,
            "raw_vi": vi_text,
            "raw_en": en_text,
            "vi_input_ids": vi_tokens,
            "vi_attention_mask": vi_attention_mask,
            "en_input_ids": en_tokens,
            "en_attention_mask": en_attention_mask,
            "source": audio_source,
        }
        
        # Cache result if enabled
        if self.use_cache:
            self.cache[idx] = result
        
        return result


@dataclass
class DataCollatorSpeechToText:
    """
    Data collator that will dynamically pad the inputs received and process audio features.
    
    Args:
        feature_extractor: SeamlessM4TFeatureExtractor for audio processing
        processor: AutoProcessor for text tokenization (used for padding)
        padding: Padding strategy for text (default: True for dynamic padding)
        pad_to_multiple_of: Pad to a multiple of this value for efficiency
        target_language: Which language to use as target/labels ("vi" or "en", default: "vi")
        pivot_language: Which language to use as pivot/context ("vi" or "en", default: "en")
        ignore_index: Value to use for padding in labels (default: -100)
    """
    
    feature_extractor: SeamlessM4TFeatureExtractor
    processor: AutoProcessor
    padding: bool = True
    pad_to_multiple_of: Optional[int] = 8
    target_language: str = "vi"  # Vietnamese as target
    pivot_language: str = "en"   # English as pivot/context
    ignore_index: int = -100
    
    def _extract_audio_features(self, waveforms: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract and pad audio features"""
        return self.feature_extractor(
            raw_speech=waveforms,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            sampling_rate=16000,
            return_attention_mask=True,
            do_normalize_per_mel_bins=True,
        )
    
    def _pad_tokens(self, token_ids_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Pad token sequences"""
        return self.processor.tokenizer.pad(
            {"input_ids": token_ids_list},
            padding=self.padding,
            return_tensors="pt"
        )
    
    def _prepare_labels(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Prepare labels by replacing padding tokens with ignore_index"""
        mask = attention_mask.eq(1)
        return torch.where(mask, token_ids, torch.tensor(self.ignore_index))
    
    def _get_language_tokens(self, features: List[Dict[str, Any]], language: str) -> List[torch.Tensor]:
        """Extract token IDs for specified language"""
        col_name = f"{language}_input_ids"
        return [f[col_name] for f in features if f[col_name] is not None]
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of samples: extract audio features and pad text tokens.
        
        Args:
            features: List of samples from ViBaSpeechToTextDataset
            
        Returns:
            Dictionary containing batched and padded features ready for model input
        """
        # Extract waveforms for audio feature extraction
        waveforms = [f["waveform"] for f in features]
        
        # Process audio features using SeamlessM4TFeatureExtractor
        audio_features = self._extract_audio_features(waveforms)
        
        batch = {
            "audio_input_features": audio_features["input_features"],
            "audio_attention_mask": audio_features["attention_mask"],
        }
        
        # Process target language tokens (labels)
        target_ids = self._get_language_tokens(features, self.target_language)
        
        if target_ids:
            labels_batch = self._pad_tokens(target_ids)
            labels = self._prepare_labels(labels_batch["input_ids"], labels_batch["attention_mask"])
            batch["labels"] = labels
        
        # Process pivot language tokens (for context/conditioning)
        pivot_ids = self._get_language_tokens(features, self.pivot_language)
        
        if pivot_ids:
            pivot_batch = self._pad_tokens(pivot_ids)
            batch["text_input_pivot_ids"] = pivot_batch["input_ids"]
            batch["text_pivot_attention_mask"] = pivot_batch["attention_mask"]
        
        return batch

