#!/usr/bin/env python3
import torch

n_time_frames = 1024 # Required by/due to the ASTModel pretraining
nmels = 128
hop_length = 512

clip_length_samples = ((n_time_frames - 1) * hop_length) + 1 # Ensures that the output of MelSpectrogramFeatureAccessor will have shape (1, nmels, n_time_frames)
clip_overlap_samples = int(clip_length_samples * 0.25)