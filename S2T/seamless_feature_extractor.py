# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Feature extractor class for SeamlessM4T
"""

import torch
import logging
import numpy as np
from typing import Optional, Union, List, Dict
import torchaudio

from enum import Enum

logger = logging.getLogger(__name__)


def is_torch_tensor(obj):
    return isinstance(obj, torch.Tensor)

class PaddingStrategy(Enum):
    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"

class TensorType(Enum):
    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"

def to_numpy(obj):
    if is_torch_tensor(obj):
        return obj.detach().cpu().numpy()
    else:
        return np.array(obj)

def to_torch_tensor(obj):
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()
    elif is_torch_tensor(obj):
        return obj.float()
    else:
        return torch.tensor(obj, dtype=torch.float32)

class BatchFeature(dict):
    def __init__(self, data: Optional[Dict] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__()
        if data is not None:
            self.update(data)

    def convert_to_tensors(self, tensor_type: Union[None, str, TensorType] = None):
        if tensor_type is None:
            return self

        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        if tensor_type == TensorType.PYTORCH:
            as_tensor = to_torch_tensor
        else:
            as_tensor = np.asarray

        for key, value in self.items():
            try:
                if isinstance(value, list):
                    if tensor_type == TensorType.PYTORCH:
                        # Handle list of tensors/arrays
                        self[key] = torch.stack([to_torch_tensor(v) for v in value])
                    else:
                        self[key] = np.array(value)
                else:
                    self[key] = as_tensor(value)
            except Exception as e:
                raise ValueError(f"Unable to create tensor: {e}")

        return self


def window_function(window_length: int, periodic: bool = False) -> torch.Tensor:
    """
    Tạo Hann window, có tùy chọn periodic, 
    sau đó pow(0.85) để chỉnh độ taper.
    """
    window = torch.hann_window(window_length, periodic=periodic)
    return window.pow(0.85)


def mel_spectrogram(
    waveform: torch.Tensor,
    frame_length: int,
    hop_length: int,
    fft_length: int,
    power: float = 2.0,
    n_mels: Optional[int] = None,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    sample_rate: int = 16000,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
    log_mel: Optional[str] = "log",
    mel_floor: float = 1.192092955078125e-07,
    preemphasis: float = 0.97,
    remove_dc_offset: bool = True,
    periodic_window: bool = False,
):
    if not torch.is_tensor(waveform):
        waveform = torch.tensor(waveform, dtype=torch.float32)

    if remove_dc_offset:
        waveform = waveform - waveform.mean()

    if preemphasis > 0.0:
        waveform = torch.cat([waveform[:1], waveform[1:] - preemphasis * waveform[:-1]])

    window = window_function(frame_length, periodic=periodic_window)

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=fft_length,
        win_length=frame_length,
        hop_length=hop_length,
        power=power,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max if f_max is not None else sample_rate // 2,
        norm=norm,
        mel_scale=mel_scale,
        center=False,
        window_fn=lambda win_length, dtype=None, layout=None, device=None, requires_grad=False: window,
    )

    spect = transform(waveform)

    if log_mel:
        spect = torch.clamp(spect, min=mel_floor)
        if log_mel == "log":
            spect = torch.log(spect)
        elif log_mel == "log10":
            spect = torch.log10(spect)

    return spect


class SeamlessM4TFeatureExtractor:
    """
    Constructs a SeamlessM4T feature extractor.

    This class extracts mel-filter bank features from raw speech.

    Args:
        feature_size (`int`, *optional*, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, *optional*, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        num_mel_bins (`int`, *optional*, defaults to 80):
            Number of Mel-frequency bins.
        padding_value (`float`, *optional*, defaults to 0.0):
            The value that is used to fill the paddin   g vectors.
        stride (`int`, *optional*, defaults to 2):
            Stride used to reshape audios from shape (batch_size,num_frames,num_mel_bins) to
            (batch_size,num_frames//stride,num_mel_bins*stride).
    """

    model_input_names = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        num_mel_bins: int = 80,
        padding_value: float = 0.0,
        stride: int = 2,
        **kwargs,
    ):

        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        self.stride = stride
        self.return_attention_mask = True

        self.padding_side = "right"
        # Use torchaudio's mel scale for filter bank
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=self.num_mel_bins,
            f_min=20.0,
            f_max=sampling_rate // 2,
            power=2.0,
            norm=None,
            mel_scale="kaldi",
            center=False,
        )

        self.window = window_function(400, periodic=False)

    @staticmethod
    def zero_mean_unit_var_norm(
        input_values: List[torch.Tensor], attention_mask: List[torch.Tensor], padding_value: float = 0.0
    ) -> List[torch.Tensor]:
        """
        Every tensor in the list is normalized to have zero mean and unit variance
        """
        if attention_mask is not None:
            attention_mask = [torch.tensor(mask, dtype=torch.int32) if not isinstance(mask, torch.Tensor) else mask for mask in attention_mask]
            normed_input_values = []

            for vector, mask in zip(input_values, attention_mask):
                length = mask.sum().item()
                normed_slice = vector.clone()
                if length > 0:
                    valid_portion = vector[:length]
                    mean_val = valid_portion.mean()
                    var_val = valid_portion.var() + 1e-7
                    normed_slice[:length] = (valid_portion - mean_val) / torch.sqrt(var_val)
                    if length < normed_slice.shape[0]:
                        normed_slice[length:] = padding_value

                normed_input_values.append(normed_slice)
        else:
            normed_input_values = [(x - x.mean()) / torch.sqrt(x.var() + 1e-7) for x in input_values]

        return normed_input_values

    def _extract_fbank_features(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get mel-filter bank features using PyTorch implementation.
        """
        # Convert to torch tensor if needed
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        
        # by default, it extracts the left channel if stereo
        if len(waveform.shape) == 2:
            waveform = waveform[0]

        waveform = waveform.squeeze() * (2**15) # Kaldi compliance: 16-bit signed integers
        # Apply mel spectrogram transform
        features = self.mel_transform(waveform).T
        
        return features

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        padding: Union[bool, str, PaddingStrategy] = True,
        pad_to_multiple_of: Optional[int] = 2,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        sampling_rate: Optional[int] = 1,
        return_attention_mask: Optional[bool] = None,
        do_normalize_per_mel_bins: Optional[bool] = True,
        **kwargs,
    ) -> BatchFeature:
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        return_attention_mask = (
            return_attention_mask if return_attention_mask is not None else self.return_attention_mask
        )

        # Convert input to appropriate format
        if isinstance(raw_speech, np.ndarray):
            if raw_speech.dtype == np.float64:
                raw_speech = raw_speech.astype(np.float32)
        elif isinstance(raw_speech, torch.Tensor):
            raw_speech = raw_speech.float()
        elif isinstance(raw_speech, list) and len(raw_speech) > 0:
            # Handle list of arrays/tensors
            if isinstance(raw_speech[0], np.ndarray):
                raw_speech = [arr.astype(np.float32) if arr.dtype == np.float64 else arr for arr in raw_speech]
            elif isinstance(raw_speech[0], torch.Tensor):
                raw_speech = [tensor.float() for tensor in raw_speech]

        # Check if input is batched
        is_batched = isinstance(raw_speech, (list, tuple)) and len(raw_speech) > 0 and isinstance(raw_speech[0], (np.ndarray, list))
        
        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # extract fbank features
        features = [self._extract_fbank_features(waveform) for waveform in raw_speech]

        if do_normalize_per_mel_bins:
            normalized_features = []
            for x in features:
                if isinstance(x, torch.Tensor):
                    mean_vals = x.mean(0, keepdim=True)
                    var_vals = x.var(0, unbiased=True ,keepdim=True) + 1e-7
                    normalized = (x - mean_vals) / torch.sqrt(var_vals)
                    normalized_features.append(normalized)
                else:
                    # Fallback to numpy
                    mean_vals = np.expand_dims(x.mean(0), 0)
                    var_vals = np.expand_dims(x.var(0, ddof=1), 0) + 1e-7
                    normalized = (x - mean_vals) / np.sqrt(var_vals)
                    normalized_features.append(normalized)
            features = normalized_features

        # convert into correct format for padding
        encoded_inputs = BatchFeature({"input_features": features})

        padded_inputs = self.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt" if return_tensors == "pt" else "np",
        )

        # SeamlessM4T needs to process extracted features
        input_features = padded_inputs.get("input_features")
        attention_mask = padded_inputs.pop("attention_mask")

        # Convert to appropriate tensor type
        if return_tensors == "pt" and isinstance(input_features, np.ndarray):
            input_features = torch.from_numpy(input_features).float()
            attention_mask = torch.from_numpy(attention_mask).long()
        elif return_tensors != "pt" and isinstance(input_features, torch.Tensor):
            input_features = input_features.detach().cpu().numpy()
            attention_mask = attention_mask.detach().cpu().numpy()

        batch_size, num_frames, num_channels = input_features.shape

        remainder = num_frames % self.stride
        if remainder != 0:
            input_features = input_features[:, : num_frames - remainder, :]
            attention_mask = attention_mask[:, : num_frames - remainder]

        if isinstance(input_features, torch.Tensor):
            input_features = input_features.reshape(
                batch_size, num_frames // self.stride, num_channels * self.stride
            )
            indices = torch.arange(0, num_frames - remainder)
            attention_mask = attention_mask[:, indices % self.stride == 1]
        else:
            input_features = np.reshape(
                input_features, (batch_size, num_frames // self.stride, num_channels * self.stride)
            )
            indices = np.arange(0, num_frames - remainder)
            attention_mask = attention_mask[:, indices % self.stride == 1]

        padded_inputs["input_features"] = input_features
        if return_attention_mask:
            padded_inputs["attention_mask"] = attention_mask

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs

    def pad(
        self,
        processed_features: Union[
            BatchFeature,
            List[BatchFeature],
            Dict[str, BatchFeature],
            Dict[str, List[BatchFeature]],
            List[Dict[str, BatchFeature]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        if isinstance(processed_features, (list, tuple)) and isinstance(processed_features[0], (dict, BatchFeature)):
            processed_features = {
                key: [example[key] for example in processed_features] for key in processed_features[0].keys()
            }

        if self.model_input_names[0] not in processed_features:
            raise ValueError(
                f"You should supply an instance of BatchFeature or list of BatchFeature "
                f"to this method that includes {self.model_input_names[0]}, but you provided "
                f"{list(processed_features.keys())}"
            )

        required_input = processed_features[self.model_input_names[0]]
        return_attention_mask = (
            return_attention_mask if return_attention_mask is not None else self.return_attention_mask
        )

        if len(required_input) == 0:
            if return_attention_mask:
                processed_features["attention_mask"] = []
            return processed_features

        first_element = required_input[0]
        if isinstance(first_element, (list, tuple)):
            index = 0
            while len(required_input[index]) == 0:
                index += 1
            if index < len(required_input):
                first_element = required_input[index][0]

        if return_tensors is None:
            return_tensors = "np"

        # Convert to appropriate tensor format
        if return_tensors == "pt":
            for key, value in processed_features.items():
                processed_features[key] = [to_torch_tensor(v) for v in value]
        else:
            for key, value in processed_features.items():
                processed_features[key] = [to_numpy(v) for v in value]

        padding_strategy = self._get_padding_strategies(padding=padding, max_length=max_length)

        required_input = processed_features[self.model_input_names[0]]

        batch_size = len(required_input)
        if not all(len(v) == batch_size for v in processed_features.values()):
            raise ValueError("Some items in the output dictionary have a different batch size than others.")

        truncated_inputs = []
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in processed_features.items())
            inputs_slice = self._truncate(
                inputs,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                truncation=truncation,
            )
            truncated_inputs.append(inputs_slice)

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = max(len(input_slice[self.model_input_names[0]]) for input_slice in truncated_inputs)
            padding_strategy = PaddingStrategy.MAX_LENGTH

        batch_outputs = {}
        for i in range(batch_size):
            outputs = self._pad(
                truncated_inputs[i],
                max_length=max_length,
                padding_strategy=padding_strategy,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                if isinstance(value, np.ndarray) and value.dtype == np.float64:
                    value = value.astype(np.float32)
                elif isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                    value = value.float()
                batch_outputs[key].append(value)

        return BatchFeature(batch_outputs, tensor_type=return_tensors)

    def _pad(
        self,
        processed_features: Dict[str, np.ndarray],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        required_input = processed_features[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) < max_length

        if return_attention_mask and "attention_mask" not in processed_features:
            processed_features["attention_mask"] = np.ones(len(required_input), dtype=np.int32)

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            if self.padding_side == "right":
                if return_attention_mask:
                    processed_features["attention_mask"] = np.pad(
                        processed_features["attention_mask"], (0, difference)
                    )
                padding_shape = ((0, difference), (0, 0)) if self.feature_size > 1 else (0, difference)
                processed_features[self.model_input_names[0]] = np.pad(
                    required_input, padding_shape, "constant", constant_values=self.padding_value
                )
            elif self.padding_side == "left":
                if return_attention_mask:
                    processed_features["attention_mask"] = np.pad(
                        processed_features["attention_mask"], (difference, 0)
                    )
                padding_shape = ((difference, 0), (0, 0)) if self.feature_size > 1 else (difference, 0)
                processed_features[self.model_input_names[0]] = np.pad(
                    required_input, padding_shape, "constant", constant_values=self.padding_value
                )
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return processed_features

    def _truncate(
        self,
        processed_features: Dict[str, np.ndarray],
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        truncation: Optional[bool] = None,
    ):
        if not truncation:
            return processed_features
        elif truncation and max_length is None:
            raise ValueError("When setting ``truncation=True``, make sure that ``max_length`` is defined.")

        required_input = processed_features[self.model_input_names[0]]

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_truncated = len(required_input) > max_length

        if needs_to_be_truncated:
            processed_features[self.model_input_names[0]] = processed_features[self.model_input_names[0]][:max_length]
            if "attention_mask" in processed_features:
                processed_features["attention_mask"] = processed_features["attention_mask"][:max_length]

        return processed_features

    def _get_padding_strategies(self, padding=False, max_length=None):
        if padding is not False:
            if padding is True:
                padding_strategy = PaddingStrategy.LONGEST
            elif not isinstance(padding, PaddingStrategy):
                padding_strategy = PaddingStrategy(padding)
            elif isinstance(padding, PaddingStrategy):
                padding_strategy = padding
        else:
            padding_strategy = PaddingStrategy.DO_NOT_PAD

        if max_length is None:
            if padding_strategy == PaddingStrategy.MAX_LENGTH:
                raise ValueError(
                    f"When setting ``padding={PaddingStrategy.MAX_LENGTH}``, make sure that max_length is defined"
                )

        if padding_strategy != PaddingStrategy.DO_NOT_PAD and (self.padding_value is None):
            raise ValueError(
                "Asking to pad but the feature_extractor does not have a padding value. Please select a value to use"
                " as `padding_value`. For example: `feature_extractor.padding_value = 0.0`."
            )

        return padding_strategy

__all__ = ["SeamlessM4TFeatureExtractor"]