# SelfSupervisedAST Directory
This directory should contain python scripts for model instantiation and dataloading for self-supervised Audio Spectrogram Transformer pre-training (train, val, test) and fine-tuning (train, val, test).

The directory should also contain a directory 'ssast', that is a git submodule set to track the original ssast implementation repository. 
This repository: https://github.com/YuanGongND/ssast
Specifically this commit (as per. 24. May 2022): main/b589c9c6eb744fe8d05340169ed36e46e8c19ba1

If not present, run this to add repo as submodule:
````bash
cd code/experiments/SelfSupervisedAST/
git submodule add https://github.com/YuanGongND/ssast
cd ./ssast 
git checkout b589c9c6eb744fe8d05340169ed36e46e8c19ba1
cd ..
````