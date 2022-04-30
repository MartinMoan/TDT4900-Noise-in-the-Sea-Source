#!/usr/bin/env python3
from dataclasses import dataclass
import multiprocessing
import pathlib
from typing import Union

from rich import print

@dataclass
class WaveFileHeader:
    filepath: Union[str, pathlib.Path]
    header: bytes
    file_type_header: str
    sample_rate: int
    n_channels: int
    num_samples: int
    duration_seconds: float
    filesize_bytes: int

class WaveFileInspector:
    def info(filepath: Union[str, pathlib.Path]) -> WaveFileHeader:
        _filepath = pathlib.Path(filepath)
        if not _filepath.exists():
            raise ValueError
        if not _filepath.is_file():
            raise ValueError
        if str(_filepath.suffix).strip().lower() != ".wav":
            raise ValueError
    
        with open(_filepath, "rb") as binaryfile:
            # based on wavefile byte table here: https://docs.fileformat.com/audio/wav/
            header = binaryfile.read(44)
            file_type_header = header[8:12].decode("utf8")
            if file_type_header != "WAVE":
                raise Exception(f"The file type header bytes suggests that the provided filepath is not a wavefile: {repr(file_type_header)}")
            
            sample_rate = int.from_bytes(header[24:28], "little", signed=False)
            bits_per_sample = int.from_bytes(header[34:36], "little", signed=False)
            size_bytes = int.from_bytes(header[4:8], "little", signed=False)
            data_chunk_size_bytes = int.from_bytes(header[40:44], "little", signed=False)
            num_samples = data_chunk_size_bytes / (bits_per_sample / 8)
            
            duration = (num_samples / sample_rate)
            n_channels = int.from_bytes(header[22:24], "little", signed=False)
            
            return WaveFileHeader(
                filepath=_filepath,
                header=header,
                file_type_header=file_type_header,
                sample_rate=sample_rate,
                n_channels=n_channels,
                num_samples=num_samples,
                duration_seconds=duration,
                filesize_bytes=size_bytes
            )
    
    def duration(filepath: Union[str, pathlib.Path]) -> float:
        _filepath = pathlib.Path(filepath)
        if not _filepath.exists():
            raise ValueError
        if not _filepath.is_file():
            raise ValueError
        if str(_filepath.suffix).strip().lower() != ".wav":
            raise ValueError
    
        with open(_filepath, "rb") as binaryfile:
            header = binaryfile.read(44)
            file_type_header = header[8:12].decode("utf8")
            if file_type_header != "WAVE":
                raise Exception(f"The file type header bytes suggests that the provided filepath is not a wavefile: {repr(file_type_header)}")
            
            sample_rate = int.from_bytes(header[24:28], "little", signed=False)
            bits_per_sample = int.from_bytes(header[34:36], "little", signed=False)
            data_chunk_size_bytes = int.from_bytes(header[40:44], "little", signed=False)
            num_samples = data_chunk_size_bytes / (bits_per_sample / 8)
            duration = (num_samples / sample_rate)
            return duration

def inf(index, length, file):
    dur = WaveFileInspector.info(file)
    print(index, length, dur)

if __name__ == "__main__":
    dataset_dir = pathlib.Path("/cluster/work/martinmoan/hdd_copy/")
    files = list(dataset_dir.glob("**/*.wav"))

    import multiprocessing
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        tasks = [pool.apply_async(inf, (i, len(files), file)) for i, file in enumerate(files)]
        results = [task.get() for task in tasks]
    


# riff = header[0:4].decode("utf8")
# size_mibs = size_bytes*1024**-2
# fmt = header[12:16].decode("utf8")
# fmt_length = int.from_bytes(header[16:20], "little", signed=False)

# sr_bps_ch_bytes = int.from_bytes(header[28:32], "little", signed=False) # (Sample Rate * BitsPerSample * Channels) / 8
# bps_ch_eight_pt_one = int.from_bytes(header[32:34], "little", signed=False) # (BitsPerSample * Channels) / 8.1 - 8 bit mono2 - 8 bit stereo/16 bit mono4 - 16 bit stereo
# data_header = header[36:40].decode("utf8")


# bits_per_sample: int
# data_chunk_size_bytes: int
# riff_header: str
# filesize_mebibytes: float
# fmt: str
# fmt_length: int
# sr_bps_ch_bytes: float
# bps_ch_eight_pt_one: int
# data_header: str