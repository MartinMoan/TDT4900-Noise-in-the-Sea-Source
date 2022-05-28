#!/usr/bin/env python3
import sys
import pathlib
from typing import List, Union, Optional
import git
import torch
import numpy as np

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
import torchaudio.transforms as transforms
from interfaces import IAugment
from datasets.initdata import create_tensorset

class SpecAugment(IAugment):
    def __init__(
        self,
        branching: 3,
        nmels: int,
        hop_length: int,
        max_time_mask_index: int = None,
        max_time_mask_seconds: float = None,
        max_mel_masks: int = None,
        sr: Union[float, int] = None
        ) -> None:
        super().__init__()
        if max_time_mask_seconds is not None and sr is None:
            raise ValueError(f"max_time_mask_seconds argument was provided, but sr was not. Cannot calculate number of indeces to mask based on max_time_mask_seconds without sr provided")
        
        time_mask_param = None
        if max_time_mask_seconds is not None:
            time_mask_param = int(max_time_mask_seconds * sr)
        else:
            if max_time_mask_index is None:
                raise ValueError(f"Neither max_time_mask_seconds nor max_time_mask_index was provided, cannot perform TimeMasking.")
            else:
                time_mask_param = max_time_mask_index

        if max_mel_masks <= 0.0 or max_mel_masks > nmels:
            raise ValueError(f"max_mel_masks has invalid value, must be in range (0, nmels]")

        self.stretch = transforms.TimeStretch(hop_length=hop_length, n_freq=nmels, fixed_rate=0.8)
        self.timemask = transforms.TimeMasking(time_mask_param=time_mask_param)
        self.freqmask = transforms.FrequencyMasking(freq_mask_param=max_mel_masks)
        self._branching = branching
        
    def _augment(self, spectrogram: torch.Tensor) -> torch.Tensor:
        print(spectrogram.dtype)
        stretched = self.stretch(spectrogram)
        
        print(stretched.dtype)
        timemasked = self.timemask(stretched)
        freqmasked = self.freqmask(timemasked)
        return freqmasked

    def forward(self, spectrogram: torch.Tensor, label: torch.Tensor) -> List[torch.Tensor]:
        if spectrogram.dim() == 3:
            # (batch or 1, nmels, time_frames)
            copies = [(self._augment(torch.clone(spectrogram)), torch.clone(label)) for i in range(self._branching)]
        elif spectrogram.dim() == 2:
            # (nmels, time_frames)
            pass
        else: 
            raise ValueError(f"Spectrogram has incorrect dimensions {spectrogram.shape}")

    def branching(self) -> int:
        return self._branching

if __name__ == "__main__":
    nfft = 3200
    nmels = 128
    hop_length = 512
    clip_duration_seconds = 10.0
    clip_overlap_seconds = 4.0

    tensorset, balancer = create_tensorset(
        nfft = nfft,
        nmels = nmels,
        hop_length = hop_length,
        clip_duration_seconds = clip_duration_seconds,
        clip_overlap_seconds = clip_overlap_seconds,
    )

    s = SpecAugment(
        branching=3,
        nmels=nmels,
        hop_length=hop_length,
        max_time_mask_seconds=2.0,
        max_mel_masks=nmels,
        sr=128000
    )

    spect, label = torch.rand((1, 128, 2048)).float(), torch.randint(0, 2, (1, 2))
    print(spect.dtype)
    augmented = s(spect, label)
    print(augmented)