#!/usr/bin/env python3
from datetime import datetime, timedelta
import sys
import pathlib
import git
import unittest
import multiprocessing
from multiprocessing.pool import ThreadPool

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from datasets.glider.clipping import ClippedDataset, CachedClippedDataset
from datasets.binjob import Binworker, progress
from datasets.tensordataset import TensorAudioDataset, MelSpectrogramFeatureAccessor, BinaryLabelAccessor

class TestTensorset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clips = CachedClippedDataset(clip_duration_seconds=10.0, clip_overlap_seconds=4.0)
        cls.tensorset = TensorAudioDataset(
            cls.clips, 
            label_accessor=BinaryLabelAccessor(), 
            feature_accessor=MelSpectrogramFeatureAccessor(
                n_mels=128,
                n_fft=3200,
                hop_length=1280
            )
        )

    def getitem(self, dataset, start, stop):
        results = []
        proc = multiprocessing.current_process()
        last_logged = datetime.now()
        log_interval = timedelta(seconds=4)
        for i in range(start, min(stop, len(dataset))):
            should_log, percentage = progress(i, start, stop)
            if should_log or datetime.now() - last_logged >= log_interval:
                print(f"ClippingWorker PID {proc.pid} - {percentage:.2f}%")
            try:
                X, Y = dataset[i]
                results.append((i, None))
            except Exception as ex:
                print(i, ex)
                results.append((i, ex))
        return results

    def test_duplicates(self):
        worker = Binworker(ThreadPool, n_processes=multiprocessing.cpu_count(), timeout_seconds=30.0)
        results = worker.apply(self.tensorset, self.getitem)
        exceptions = [el for el in results if el[1] is not None]
        self.assertEqual(len(exceptions), 0, f"There were exceptions raised during loading: {exceptions}")

if __name__ == "__main__":
    unittest.main()