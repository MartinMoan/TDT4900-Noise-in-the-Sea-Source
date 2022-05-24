# TDT4900 Noise in the Sea Source
This repository contains the source code for the [Noise in the Sea masters project](https://github.com/MartinMoan/TDT4900-Noise-in-the-Sea). Meaning python code to parse the GLIDER dataset labels, loading the GLIDER dataset itself with labeled and unlabeled examples, pytorch model implementations and any other source code related to the project datasets and models. Source code/markup/tex-files that are relevant to the project thesis can be found [here](https://github.com/MartinMoan/TDT4900-Noise-in-the-Sea)

````
On Cluster:
clip_duration_seconds=10.0, 
clip_overlap_seconds=4.0
number of clips: 470149
````

- [x] TODO: Use Metric collections rather than individually called metrics
- [x] TODO: Add mAP (NOT for each label, but averaged across classes) to metriccollection
- [ ] TODO: Check that mAP works correctly using GPU(s) and/or Nodes
- [x] TODO: Check that logging metriccollection works correctly using WandbLogger
- [ ] TODO: Check that logging mAP (in/without metriccollection) works as expected
- [ ] TODO: Check where model parameters are stored, and verify that I can reinstantiate a trained model for future use. This is for reproducibility. 
- [ ] TODO: Implement model.predict() from LightningModule? Check what this method is for, how to use it, and why I should/should not use it.
- [ ] TODO: Ensure that the path to the stored parameters, or the parameter file itself, is logged correctly. Such that I can select the best model from my wandb logs/plots and instantiate that exact model easily.
- [ ] TODO: Store raw predictions and targets in testing? Such that I can compute metrics multiple times after model is trained?
- [ ] TODO: Rerun AST training. Run using 4/5 GPUs (seems 6 GPUs take a long time to be scheduled) and on 1 Node (IMPORTANT!!) (becuase using multiple nodes when not needed makes training less efficient). Try 24 CPUs and ensure that dataloaders use this number of workers.
- [ ] TODO: Implement SSAST training, validatino and testing
- [ ] TODO: Implement learning rate scheduling (as described in AST, and SSAST (when I have implemented SSAST training))
- [ ] TODO: Implement logging a few raw batches, to verify that the actual tensors/data passed to the model during training,val and testing are valid/correct/as expected. 
- [ ] TODO: Implement on_save_checkpoint and on_load_checkpoint pytorch_lightning callbacks to ensure that the WANDB_RUN_ID is stored with the checkpoint. Such that when crashed runs are restarted, they use the same run_id as the crashed run and logs are sent to the same run in Wandb. Currently, if a run crashed, it is restarted (successfully) from an automatically created checkpoint, and training/validation/testing resumes as expected, but the WandbLogger sends logged data (typically metrics) to a new run rather than the crashed one.

Future work:
- [ ] Make DataModule a standalone feature/script and make it "completely" reusable (with optional clipping, make it possible but mandatory that the datamodule download the dataset from the Azure blobs automatically). Maybe even its own project/package that can be installed? Would be cool, and make working with the glider audio more efficient and more reproducible.