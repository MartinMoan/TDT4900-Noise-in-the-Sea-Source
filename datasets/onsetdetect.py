#!/usr/bin/env python3
from cProfile import label
import sys
import pathlib
from typing import Mapping, Iterable

import git
import librosa
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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
    proc = play.play(reduced, sr)
    proc.wait()

class LabelErrorCorrector(ICustomDataset):
    def __init__(self, decorated: ICustomDataset, logger_factory: ILoggerFactory, worker: IAsyncWorker, ) -> None:
        super().__init__()
        self._decorated = decorated
        self.logger = logger_factory.create_logger()
        self.worker = worker
        self._procs = []
        self.correct_label_mismatch()
    
    def _play(self, samples, sr):
        def run(*args):
            if len(self._procs) != 0:
                self._stop()
            else:
                proc = play.play(samples, sr)
                self._procs.append(proc)
        return run

    def _stop(self):
        for i, (proc) in enumerate(self._procs):
            proc.terminate()
            del self._procs[i]

    def plot(self, labeled_audiodata: LabeledAudioData) -> None:
        plt.switch_backend('TkAGG')
        spectcomputer = MelSpectrogramFeatureAccessor()

        samples, sr = labeled_audiodata.samples, labeled_audiodata.sampling_rate
        spect = spectcomputer(labeled_audiodata)
        rows = 6
        
        start, end = labeled_audiodata.start_time, labeled_audiodata.end_time
        reduced = nr.reduce_noise(y=samples, sr=sr, prop_decrease=1.0)

        axes = plt.axes([0.01, 1.0-(1.0/rows), 0.1, 0.05])
        playbutton = Button(axes, "Play")
        playbutton.on_clicked(self._play(samples, sr))
        
        axes2 = plt.axes([0.01,  0.15, 0.1, 0.05])
        pl2 = Button(axes2, "Play filtered")
        pl2.on_clicked(self._play(reduced, sr))

        classes = labeled_audiodata.labels.source_class_specific.unique()
        c = ", ".join(classes)
        plt.suptitle(f"{labeled_audiodata.filepath}\n{start} {end}\n{c}")
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

        S = spectcomputer._compute(reduced, sr)
        plt.imshow(S.squeeze(), aspect="auto")
        # plt.tight_layout()

        figM = plt.get_current_fig_manager()
        figM.full_screen_toggle()
        
        plt.show()
        self._stop()

        self.logger.log(f"i: {labeled_audiodata._index}, n samples: {len(samples)}, sr: {sr}")

    def correct_label_mismatch(self):
        noise_file = config._GLIDER_DATASET_DIRECTORY.joinpath("August", "pa0313au_001_180420_235932.wav")
        # noise_start_sec, noise_end_sec = 530, 580
        # noise, noise_sr = librosa.load(noise_file, sr=None, offset=noise_start_sec, duration=(noise_end_sec - noise_start_sec))
        
        for i in range(100, len(self._decorated)):
            labeled_audiodata = self._decorated[i]
            print(i, labeled_audiodata.labels.source_class_specific)
            if len(labeled_audiodata.labels.source_class_specific.unique()) != 0:
                self.plot(labeled_audiodata)
            

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
        clip_duration_seconds = 10.0,
        clip_overlap_seconds = 3.0
    )

    balancer = CachedDatasetBalancer(
        clip_dataset,
        logger=logger,
        worker=worker,
        force_recache=False
    )

    limited_dataset = DatasetLimiter(
        clip_dataset, 
        limit=200, 
        randomize=False, 
        balanced=True,
        balancer=balancer
    ) # If randomize=True, the transform will never be cahced (because the fitted dataset changes between sessions, due to randomization)

    corrected = LabelErrorCorrector(
        limited_dataset,
        logger_factory=factory,
        worker=worker
    )
