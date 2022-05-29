#!/usr/bin/env python3
import sys
import pathlib
from typing import List, Union, Optional, Tuple, Iterable
import git
import torch
import numpy as np

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))

import torch
import torch.linalg
import torchaudio.transforms as transforms
from interfaces import IAugment
from datasets.initdata import create_tensorset
import matplotlib.pyplot as plt
from sparse_image_warp import time_warp

def show_spect(spects: Iterable[torch.Tensor]):
    N = len(spects)
    for n, spect in enumerate(spects):
        plt.subplot(N, 1, n + 1)
        plt.imshow(spect.squeeze(), aspect="auto", cmap="magma_r")
    plt.show()

class CombinedAugment(IAugment):
    def __init__(self, *augments) -> None:
        super().__init__()
        self.augments = augments
        self._branching = np.prod([augment.branching() for augment in augments]) if len(augments) > 0 else 0

    def forward(self, spectrogram: torch.Tensor, label: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        queue = [(spectrogram, label)]
        buffer = []
        for i, augment in enumerate(self.augments):
            while len(queue) > 0:
                spect, label = queue.pop(0)
                buffer += augment(spect, label)
            queue = buffer
            buffer = []
        return queue
        
    def branching(self) -> int:
        return self._branching

class GaussianAugment(IAugment):
    def __init__(self, mean: Optional[Union[float, int]] = 0.0, std: Optional[Union[float, int]] = 1.0) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, data: torch.Tensor, label: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        r = torch.normal(self.mean, self.std, size=data.size())
        return [((data + r), label)]

    def branching(self) -> int:
        return 1

class SpecAugment(IAugment):
    def __init__(
        self,
        branching: 3,
        nmels: int,
        hop_length: int,
        max_time_mask_index: int = None,
        max_time_mask_seconds: float = None,
        max_mel_masks: int = None,
        sr: Union[float, int] = None,
        time_warp_w_parameter: Optional[int] = 80, # 80 as per original SpecAugment paper.
        max_fails: Optional[int] = 80,
        # output_originals: Optional[bool] = False
        ) -> None:
        super().__init__()
        if max_time_mask_seconds is not None and sr is None:
            raise ValueError(f"max_time_mask_seconds argument was provided, but sr was not. Cannot calculate number of indeces to mask based on max_time_mask_seconds without sr provided")
        
        time_mask_param = None
        if max_time_mask_seconds is not None:
            time_mask_param = int((max_time_mask_seconds * sr / hop_length) + 1)
        else:
            if max_time_mask_index is None:
                raise ValueError(f"Neither max_time_mask_seconds nor max_time_mask_index was provided, cannot perform TimeMasking.")
            else:
                time_mask_param = max_time_mask_index

        if max_mel_masks <= 0.0 or max_mel_masks > nmels:
            raise ValueError(f"max_mel_masks has invalid value, must be in range (0, nmels]")

        self.time_warp_w_parameter = time_warp_w_parameter
        self.timemask = transforms.TimeMasking(time_mask_param=time_mask_param)
        self.freqmask = transforms.FrequencyMasking(freq_mask_param=max_mel_masks)
        self._branching = branching
        self._max_atempts = max_fails
        self._n_fails = 0
        self._caught_exceptions = []
        # self.output_originals = output_originals
        
    def _augment(self, spectrogram: torch.Tensor) -> torch.Tensor:
        output = spectrogram
        try:
            # time_warp sometimes fails due to non-invertible input in internals
            # dont know why, but will allow N number of time_warp tries to fail and in those cases just perform time- and frequency masking without time_warping
            output = time_warp(spectrogram, W=self.time_warp_w_parameter)
        except Exception as ex:
            self._n_fails += 1
            self._caught_exceptions.append(ex)
            if self._n_fails >= self._max_atempts:
                raise Exception(f"Too many exceptions has occured when performing specaugment: {str(self._caught_exceptions)}")
            
        output = self.timemask(output)
        output = self.freqmask(output)
        assert output.shape == spectrogram.shape
        return output

    def forward(self, normalized_spectrogram: torch.Tensor, label: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Input a single normalized, spectrogram corresponding and label, and outputs a list of N "virtual" examples. Where N == self.branching. The target is replicated equally (e.g. unchanged for all of the new samples)

        Args:
            normalized_spectrogram (torch.Tensor): A spectrogram normalized to have zero mean and unit deviation computed in relation to the training set. 
            label (torch.Tensor): The target label(s) for the spectrogram.

        Raises:
            ValueError: _description_

        Returns:
            List[torch.Tensor]: _description_
        """
        if normalized_spectrogram.dim() == 3:
            # (batch, nmels, time_frames)
            copies = [(self._augment(torch.clone(normalized_spectrogram)), torch.clone(label)) for i in range(self._branching)]
            # return [(normalized_spectrogram, label), *copies] if self.output_originals else copies
            return copies
        else: 
            raise ValueError(f"Spectrogram has incorrect dimensions {normalized_spectrogram.shape}")

    def branching(self) -> int:
        # return self._branching + 1 if self.output_originals else self._branching
        return self._branching

if __name__ == "__main__":
    nfft = 3200
    nmels = 128
    hop_length = 512
    clip_duration_seconds = 10.0
    clip_overlap_seconds = 4.0
    branching = 3

    tensorset, balancer = create_tensorset(
        nfft = nfft,
        nmels = nmels,
        hop_length = hop_length,
        clip_duration_seconds = clip_duration_seconds,
        clip_overlap_seconds = clip_overlap_seconds,
    )

    s = SpecAugment(
        branching=branching,
        nmels=nmels,
        hop_length=hop_length,
        max_time_mask_seconds=2.0,
        max_mel_masks=nmels // 8,
        sr=128000
    )

    augments = CombinedAugment(s, GaussianAugment())

    for i in range(5):
        spect, label = tensorset[i]
        augmented = augments(spect, label)
        spects = [spect for spect, _ in augmented]
        show_spect([spect, *spects])