#!/usr/bin/env python3
from datetime import datetime, timedelta
import pathlib
import pickle
import sys
import re
import multiprocessing
from typing import Iterable

from rich import print
import pandas as pd
import numpy as np
import git
import librosa

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config
from interfaces import IAsyncWorker, ILoggerFactory, IAudioFileInfoProvider
from cacher import Cacher
from datasets.binjob import progress
from datasets.glider.wavfileinspector import WaveFileHeader, WaveFileInspector

class AudioFileInfoProvider(IAudioFileInfoProvider):
    def __init__(self, filelist: Iterable[pathlib.Path], logger_factory: ILoggerFactory, worker: IAsyncWorker) -> None:
        super().__init__()
        self.logger = logger_factory.create_logger()
        self.worker = worker
        self.filelist = np.sort(filelist, axis=0)
        self.cacher = Cacher(logger_factory=logger_factory)

    def info(self, files: Iterable, start: int, end: int) -> Iterable[pathlib.Path]:
        proc = multiprocessing.current_process()
        output = []
        last_logged_at = datetime.now()
        for i in range(start, min(end, len(files))):
            should_log, percentage = progress(i, start, end)
            if should_log or (datetime.now() - last_logged_at) >= timedelta(seconds=config.LOG_INTERVAL_SECONDS):
                self.logger.log(f"{self.__class__.__name__}Worker PID {proc.pid} - {percentage:.2f}%")
            
            info = WaveFileInspector.info(files[i])
            output.append(info)
        return output

    def files(self) -> pd.DataFrame:
        cache_dir = config.CACHE_DIR.joinpath("files")
        if not cache_dir.exists():
            cache_dir.mkdir(parents=False, exist_ok=False)

        pickle_path = self.cacher.hash(cache_dir, {"filelist": self.filelist})

        if not pickle_path.exists():
            results = self.worker.apply(self.filelist, self.info)

            data = []
            for file in results:
                filename_information = re.search(r"([^_]+)_([0-9]{3})_([0-9]{6}_[0-9]{6})\.wav", file.filepath.name)
        
                # dive_number = filename_information.group(1)
                # identification_number = filename_information.group(2)
                timestring = filename_information.group(3)
                start_time = None
                try:
                    start_time = datetime.strptime(timestring, "%y%m%d_%H%M%S")
                except Exception as ex:
                    print(ex)
                    start_time = datetime(year=1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
                
                try:
                    samples, sr = librosa.load(file.filepath, sr=None)
                except:
                    continue

                item = {
                    'filename': file.filepath,
                    'num_channels': file.n_channels,
                    'sampling_rate'  : file.sample_rate,
                    'num_samples': int(file.num_samples),
                    'duration_seconds': file.duration_seconds,
                    "start_time": start_time,
                    "end_time": start_time + timedelta(seconds=file.duration_seconds)                
                }
                data.append(item)
            
            df = pd.DataFrame(columns=['filename', 'num_channels', 'sampling_rate', 'num_samples', 'duration_seconds', 'start_time', 'end_time'], data=data)
            with open(pickle_path, "wb") as binaryfile:
                pickle.dump(df, binaryfile)
            return df
        else:
            with open(pickle_path, "rb") as binaryfile:
                df = pickle.load(binaryfile)
                return df

if __name__ == "__main__":
    from tracking.logger import Logger, LogFormatter
    from tracking.loggerfactory import LoggerFactory
    from datasets.binjob import Binworker
    from datasets.glider.filelist import AudioFileListProvider
    import numpy as np

    factory = LoggerFactory(
        logger_type=Logger,
        logger_args=(LogFormatter(),)
    )

    worker = Binworker()

    filelistprovider = AudioFileListProvider()
    filelist = filelistprovider.list()
    np.random.shuffle(filelist)

    fileinfoprovider = AudioFileInfoProvider(
        logger_factory=factory,
        worker=worker,
        filelist=filelist
    )

    files = fileinfoprovider.files()
    print(files)