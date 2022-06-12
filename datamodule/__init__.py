import pathlib
import pandas as pd

GLIDER_RECORDINGS_PATH = pathlib.Path(__file__).parent.joinpath("metadata", "metadata.csv")
GLIDER_LABELS_PATH = pathlib.Path(__file__).parent.joinpath("metadata", "labels.csv")

GLIDER_RECORDINGS = pd.read_csv(GLIDER_RECORDINGS_PATH)
GLIDER_LABELS = pd.read_csv(GLIDER_LABELS_PATH)

recordings = GLIDER_RECORDINGS
labels = GLIDER_LABELS