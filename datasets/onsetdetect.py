#!/usr/bin/env python3
import sys
import pathlib
from typing import Mapping, Iterable

import git
import librosa
import matplotlib.pyplot as plt
import numpy as np
import noisereduce as nr

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import ICustomDataset, ILoggerFactory, IAsyncWorker
from datasets.glider.audiodata import LabeledAudioData
from datasets.tensordataset import MelSpectrogramFeatureAccessor
from tools.player import play
import multiprocessing

def p(samples, reduced, sr):
    play.play(samples, sr)
    play.play(reduced, sr)

class LabelErrorCorrector(ICustomDataset):
    def __init__(self, decorated: ICustomDataset, logger_factory: ILoggerFactory, worker: IAsyncWorker, ) -> None:
        super().__init__()
        self._decorated = decorated
        self.logger = logger_factory.create_logger()
        self.worker = worker
        self.correct_label_mismatch()

    def correct_label_mismatch(self):
        spectcomputer = MelSpectrogramFeatureAccessor()
        noise_file = config._GLIDER_DATASET_DIRECTORY.joinpath("August", "pa0313au_001_180420_235932.wav")
        noise_start_sec, noise_end_sec = 530, 580
        noise, noise_sr = librosa.load(noise_file, sr=None, offset=noise_start_sec, duration=(noise_end_sec - noise_start_sec))
        
        for i in range(5, len(self._decorated)):
            labeled_audiodata = self._decorated[i]
            print(labeled_audiodata.clip_duration)
            samples, sr = labeled_audiodata.samples, labeled_audiodata.sampling_rate
            spect = spectcomputer(labeled_audiodata)
            rows = 6
            plt.subplot(rows, 1, 1)
            plt.imshow(spect.squeeze(), aspect="auto")
            
            plt.subplot(rows, 1, 2)
            onset_envelope = librosa.onset.onset_strength(y=samples, sr=sr)
            ts = librosa.times_like(onset_envelope, sr=sr)
            plt.plot(ts, onset_envelope, label="Onset strength")
            plt.xlim(0, max(ts))
            plt.legend()

            plt.subplot(rows, 1, 3)
            rms = librosa.feature.rms(y=samples)[0]
            times = librosa.times_like(rms, sr=sr)
            plt.semilogy(times, rms, label="RMS")
            plt.xlim(0, max(times))
            plt.legend()

            plt.subplot(rows, 1, 4)
            zcr = librosa.feature.zero_crossing_rate(y=samples)[0]
            times = librosa.times_like(zcr, sr=sr)
            plt.plot(times, zcr, label="Zero-crossing-rate")
            plt.xlim(0, max(times))
            plt.legend()

            plt.subplot(rows, 1, 5)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=samples, sr=sr)
            times = librosa.times_like(spectral_bandwidth, sr=sr)
            plt.semilogy(times, spectral_bandwidth[0], label="Spectral bandwidth")
            plt.xlim(0, max(times))
            plt.legend()

            plt.subplot(rows, 1, 6)

            reduced = nr.reduce_noise(y=samples, sr=sr, y_noise = noise)
            S = spectcomputer._compute(reduced, sr)
            plt.imshow(S.squeeze(), aspect="auto")

            plt.tight_layout()
            proc = multiprocessing.Process(target=p, args=(samples, reduced, sr))
            proc.start()
            plt.show()
            proc.join()
            exit()
            self.logger.log(f"i: {i}, n samples: {len(samples)}, sr: {sr}")

    def __getitem__(self, index: int) -> LabeledAudioData:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return super().__len__()

    def classes(self) -> Mapping[str, int]:
        return super().classes()

    def example_shapes(self) -> Iterable[tuple[int, ...]]:
        return super().example_shapes()

    def __repr__(self) -> str:
        raise NotImplementedError


if __name__ == "__main__":
    from datasets.binjob import Binworker
    from tracking.logger import Logger
    from datasets.balancing import CachedDatasetBalancer, DatasetBalancer
    from datasets.glider.clipping import CachedClippedDataset
    from datasets.limiting import DatasetLimiter
    from tracking.loggerfactory import LoggerFactory
    import matplotlib.pyplot as plt

    n_mels = 128

    factory = LoggerFactory(logger_type=Logger)
    worker = Binworker(timeout_seconds=120)
    logger = Logger()

    clip_dataset = CachedClippedDataset(
        worker=worker,
        logger=logger,
        clip_duration_seconds = 10,
        clip_overlap_seconds = 4.0
    )

    balancer = CachedDatasetBalancer(
        clip_dataset,
        logger=logger,
        worker=worker,
        force_recache=False
    )

    limited_dataset = DatasetLimiter(
        clip_dataset, 
        limit=100, 
        randomize=False, 
        balanced=True,
        balancer=balancer
    ) # If randomize=True, the transform will never be cahced (because the fitted dataset changes between sessions, due to randomization)

    corrected = LabelErrorCorrector(
        limited_dataset,
        logger_factory=factory,
        worker=worker
    )
