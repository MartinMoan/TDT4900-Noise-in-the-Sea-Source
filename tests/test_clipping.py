#!/usr/bin/env python3
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

class TestClipping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clips = CachedClippedDataset(clip_duration_seconds=10.0, clip_overlap_seconds=4.0)

    # def test_duplicates(self):
    #     data = []
    #     for i in tqdm(range(len(self.clips))):
    #         clip = self.clips[i]
    #         data.append([clip.filepath, clip.clip_offset, clip.clip_duration])
    #     df = pd.DataFrame(data=data, columns=["filepath", "offset", "duration"])

    #     duplicated = df.duplicated(subset=["filepath", "offset", "duration"], keep="first")
    #     duplicates = df.loc[duplicated]
        
    #     self.assertEqual(len(duplicates), 0, f"{len(duplicates)} duplicate clips was found in ClippedDataset with length {len(self.clips)}")

    def load_samples(self, i):
        # print(i)
        clip = self.clips[i]
        try:
            samples = clip.samples
            return (i, clip, None)
        except Exception as ex:
            return (i, clip, ex)

    def test_loads(self):
        with ThreadPool(processes=multiprocessing.cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.map(
                        func=self.load_samples, 
                        iterable=range(len(self.clips)), 
                        chunksize=len(self.clips) // multiprocessing.cpu_count()
                    ), 
                    total=len(self.clips)
                )
            )
        # results = []
        # for i in tqdm(range(len(self.clips))):
        #     results.append(self.load_samples(i))
        exceptions = [el for el in results if el[2] is not None]
        self.assertEqual(len(exceptions), 0, f"There were exceptions raised when trying to read samples: {exceptions}")

if __name__ == "__main__":
    unittest.main()