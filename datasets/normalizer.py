#!/usr/bin/env python3
from email.mime import audio
import sys
import pathlib
import abc
from typing import Mapping, Iterable, Union, Tuple
import git
from matplotlib.pyplot import isinteractive
from sklearn.preprocessing import scale
import torch
import numpy as np
import multiprocessing

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from ITensorAudioDataset import ITensorAudioDataset, BinaryLabelAccessor, MelSpectrogramFeatureAccessor, ITensorAudioDataset, TensorAudioDataset
from glider.audiodata import LabeledAudioData 
from logger import ILogger, Logger
from binjob import Binworker, progress
from limiting import DatasetLimiter
from clipping import ClippingCacheDecorator
from cacher import Cacher
from audiodata import AudioData
import warnings

TRANSFORMER_CACHE_DIR = config.CACHE_DIR.joinpath("normalization")
if not TRANSFORMER_CACHE_DIR.exists():
    TRANSFORMER_CACHE_DIR.mkdir(parents=True, exist_ok=False)

class IScaler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def transform(self, index: int, X: torch.Tensor, Y: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class ITransformer(IScaler, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, dataset: ITensorAudioDataset) -> IScaler:
        raise NotImplementedError

class StandardScalerTransformer(ITransformer):
    def __init__(self, logger: ILogger = Logger(), verbose: bool = False) -> None:
        self.logger = logger
        self.verbose = verbose
        
        self._fitted = False
        self._mean = None
        self._stddev = None

    def _get_xs(self, dataset: ITensorAudioDataset, start: int, end: int) -> Iterable:
        xs = []
        for i in range(start, end):
            should_log, progression = progress(i, start, end, log_interval_percentage=0.05)
            if should_log:
                proc = multiprocessing.current_process()
                self.logger.log(f"{self.__class__.__name__}Worker PID {proc.pid} - {progression:.2f}%")

            index, X, Y = dataset[i]
            xs.append(X.numpy())
        return xs

    def _aggregate_xs(self, xs: Iterable[Iterable[np.ndarray]]) -> torch.Tensor:
        values = np.concatenate(xs, axis=0)
        return values

    def fit(self, dataset: ITensorAudioDataset) -> IScaler:
        if not self._fitted:
            worker = Binworker()
            xs = worker.apply(dataset, self._get_xs, aggregation_method=self._aggregate_xs)

            self._mean = np.mean(xs, axis=None) # Compute over flattened array
            self._stddev = np.std(xs, axis=None) # Compute over flattened array
            self._fitted = True

            self.logger.log(f"{self.__class__.__name__} computed mean {self._mean} and standard deviation {self._stddev} of the dataset features.")
            self.logger.log(f"All features has shape {xs.shape} with dataset size {len(dataset)}")
        else:
            self.logger.log(f"{self.__class__.__name__} is already fitted when .fit(dataset) was called, skipping.")

        return self

    def transform(self, index: int, X: torch.Tensor, Y: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        if not self._fitted:
            raise Exception(f"The {ITransformer} has not been fitted yet, cannot transform before fitting. Call transformer.fit(dataset) before transform.")
        
        self.logger.log(f"Scaling data (index: {index}, X.shape: {X.shape}, Y.shape: {Y.shape}) to zero mean and unit variance, using mean {self._mean} and standard deviation {self._stddev}")
        scaled_X = (X - self._mean) / self._stddev # Same implementation as sklearn.preprocessing.StandardScaler
        return (index, scaled_X, Y)

    def __repr__(self):
        return f"{self.__class__.__name__}(_fitted={self._fitted}, _mean={self._mean}, _stddev={self._stddev})"

class CahcedStandardScalerTransformerDecorator(ITransformer):
    def __init__(self, decorated: StandardScalerTransformer = None, force_recache: bool = False, **kwargs):
        if decorated is None:
            decorated = StandardScalerTransformer(**kwargs)
        self._decorated = decorated
        self._force_recache = force_recache

    def fit(self, dataset: ITensorAudioDataset) -> IScaler:
        hashable_arguments = repr((self._decorated, dataset)).encode()
        cacher = Cacher()
        scaler = cacher.cache(TRANSFORMER_CACHE_DIR, StandardScalerTransformer, hashable_arguments=hashable_arguments, force_recache=self._force_recache)
        self._decorated = scaler

        if self._decorated._fitted:
            return self._decorated
        
        self._decorated = self._decorated.fit(dataset)
        pickle_path = cacher.hash(TRANSFORMER_CACHE_DIR, hashable_arguments=hashable_arguments)
        cacher.dump(self._decorated, pickle_path)
        return self._decorated

    def transform(self, index: int, X: torch.Tensor, Y: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        return self._decorated.transform(index, X, Y)

class TransformedTensorDataset(ITensorAudioDataset):
    def __init__(
        self, 
        decorated: ITensorAudioDataset, 
        transforms: Iterable[Union[ITransformer, IScaler]], 
        logger: ILogger = Logger(), 
        verbose: bool = False) -> None:
        super().__init__()
        self._decorated = decorated
        self.logger = logger
        self.verbose = verbose
        self.transforms = []
        for scaler in transforms:
            if isinstance(scaler, ITransformer):
                transform = scaler.fit(self._decorated)
                self.transforms.append(transform)
            elif isinstance(scaler, IScaler):
                self.transforms.append(scaler)
            else:
                raise ValueError(f"One of the provided transforms argument values has incorrect type, expected {ITransformer} or {IScaler} but received {type(scaler)}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._decorated.__getitem__(index)
        for scaler in self.transforms:
            data = scaler.transform(*data)
        return data

    def __len__(self) -> int:
        return self._decorated.__len__()

    def example_shape(self) -> tuple[int, ...]:
        return self._decorated.example_shape()

    def label_shape(self) -> tuple[int, ...]:
        return self._decorated.label_shape()

    def classes(self) -> Mapping[str, int]:
        return self._decorated.classes()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(_decorated={repr(self._decorated)}, transforms={repr(self.transforms)})"

    def audiodata(self, index: int) -> LabeledAudioData:
        return self._decorated.audiodata(index)

if __name__ == "__main__":
    n_mels = 128
    clip_dataset = ClippingCacheDecorator(
        clip_duration_seconds = 60,
        clip_overlap_seconds = 20
    )

    limited_dataset = DatasetLimiter(clip_dataset, limit=100, randomize=False, balanced=True) # If randomize=True, the transform will never be cahced (because the fitted dataset changes between sessions, due to randomization)
    limited_tensordatataset = TensorAudioDataset(
        dataset = limited_dataset,
        label_accessor=BinaryLabelAccessor(),
        feature_accessor=MelSpectrogramFeatureAccessor(n_mels=n_mels)
    )
    
    scaled_dataset = TransformedTensorDataset(limited_tensordatataset, transforms=[CahcedStandardScalerTransformerDecorator(force_recache=False)], verbose=True)

    import matplotlib.pyplot as plt

    EXAMPLES_DIR = config.HOME_PROJECT_DIR.joinpath("examples")
    if not EXAMPLES_DIR.exists():
        EXAMPLES_DIR.mkdir(parents=False, exist_ok=False)

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1 + (hz / 700))

    for i in range(len(scaled_dataset)):
        index, X, Y = scaled_dataset[i]
        audiodata = scaled_dataset.audiodata(i)

        start_sec = (audiodata.start_time - audiodata.file_start_time).seconds
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        scale = 0.8
        fig = plt.figure(figsize=(1920*scale*px, 1080*scale*px))
        plt.imshow(X.squeeze(), cmap="viridis", aspect="auto", extent=(start_sec, start_sec + audiodata.clip_duration, 0, hz_to_mel(audiodata.sampling_rate / 2)))
        plt.title(audiodata.filepath)
        plt.ylabel("Mel Frequency")
        plt.xlabel("Time after recording start [sec]")
        image_name = f"{audiodata.filepath.stem}_start_{start_sec}_end_{(start_sec + int(audiodata.clip_duration))}.png"
        store_dir = EXAMPLES_DIR.joinpath(image_name)
        print(store_dir)
        plt.tight_layout()
        plt.savefig(store_dir) 
        plt.close()