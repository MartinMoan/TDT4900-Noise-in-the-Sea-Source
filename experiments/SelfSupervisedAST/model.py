#!/usr/bin/env python3
from datetime import datetime
import pathlib
import sys
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Tuple, Union, Dict

import git
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchmetrics

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
import config

from experiments.SelfSupervisedAST.ssast.src.models import ASTModel
from metrics import customwandbplots
from metrics.metriccollection import Average, GliderMetrics

MODEL_SAVE_PATH = config.DEFAULT_PARAMETERS_PATH.joinpath("SSAST")
if not MODEL_SAVE_PATH.exists():
    MODEL_SAVE_PATH.mkdir(parents=False, exist_ok=False)

class SSASTModelSize(Enum):
    tiny = "tiny"
    small = "small"
    base = "base"
    base_nokd = "base_nokd"

class TrainingStage(Enum):
    pretrain = "pretrain"
    finetune = "finetune"

class ClipRepresentationMethod(Enum):
    average_tokens = "ft_avgtok"
    cls_tokens = "ft_cls"

class SSASTLightningWrapper(pl.LightningModule):
    def __init__(
        self, 
        learning_rate: float,
        weight_decay: float = 5e-7,
        betas: Tuple[float, float] = (0.95, 0.999),
        stage: Optional[str] = TrainingStage.pretrain.value,
        fstride: Optional[int] = 16,
        tstride: Optional[int] = 16,
        fshape: Optional[int] = 16,
        tshape: Optional[int] = 16,
        input_fdim: Optional[int] = 1024,
        input_tdim: Optional[int] = 128,
        n_model_outputs: int = 2,
        model_size: Optional[str] = SSASTModelSize.base.value,
        pretext_masked_patches: Optional[int] = 400,
        generative_loss_weight: Optional[Union[float, int]] = 10.0,
        clip_representation: Optional[str] = ClipRepresentationMethod.average_tokens.value,
        class_names: Optional[Iterable[str]] = None, 
        pretrained_model_path: Optional[Union[pathlib.Path, str]] = None,
        *args: Any, **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.stage = stage
        self.n_model_outputs = n_model_outputs
        self.fshape = fshape
        self.tshape = tshape
        self.fstride = fstride
        self.tstride = tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.ast_size = model_size
        self.pretext_masked_patches = pretext_masked_patches
        self.generative_loss_weight = generative_loss_weight
        self.clip_representation = clip_representation
        self.class_names = class_names

        self.model = self._instantiate_model(model_path=pretrained_model_path)
        # During fine-tuning stage, we are preforming multi-label classification
        # Therefore we need each output of the MPL head to yield values in the range [0,1], that do not (necessarily) sum to 1. 
        self.val_pretext_batch_accuracy = Average()
        self.test_pretext_batch_accuracy = Average()

        self.finetune_loss = torch.nn.BCELoss()
        self.finetune_activation = torch.nn.Sigmoid()

        self.val_finetune_metrics = GliderMetrics(num_classes=self.n_model_outputs, class_names=self.class_names)
        self.test_finetune_metrics = GliderMetrics(num_classes=self.n_model_outputs, class_names=self.class_names)
        
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=self.n_model_outputs, multilabel=True, threshold=0.5)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(num_classes=self.n_model_outputs, multilabel=True, threshold=0.5)

        self.save_hyperparameters()
        now = datetime.now().strftime(config.DATETIME_FORMAT)
        self.pretrained_model_filename = f"ssast-pretrained-{now}.pth"
        self.pretrain_complete = False

    def _instantiate_model(self, model_path: Optional[Union[pathlib.Path, str]] = None) -> None:
        path = None
        if model_path is not None:
            if not isinstance(model_path, (str, pathlib.Path)):
                raise TypeError(f"argument model_path has incorrect type, expected str or pathlib.Path but received object with type {type(model_path)}") 
            path = model_path.absolute() if isinstance(model_path, pathlib.Path) else model_path
           
        return ASTModel(
            label_dim=self.n_model_outputs,
            fshape=self.fshape,
            tshape=self.tshape,
            fstride=self.fstride,
            tstride=self.tstride,
            input_fdim=self.input_fdim,
            input_tdim=self.input_tdim,
            model_size=self.ast_size,
            pretrain_stage=self.is_pretrain_stage,
            load_pretrained_mdl_path=path
        )

    def pretrain(self) -> None:
        """Sets model internal state to train for pretext tasks"""
        self.stage = TrainingStage.pretrain.value
        self.model = self._instantiate_model(model_path=None)

    def save_model(self, model_path: pathlib.Path) -> None:
        torch.save(self.model.state_dict(), model_path)

    def finetune(self, pretrained_model_path: Union[pathlib.Path, str] = None) -> None:
        """Sets model internal state to train for fine-tuning task"""
        # (Mostly) following guide for SSAST usage here: https://github.com/YuanGongND/ssast
        self.save_model(pretrained_model_path)
        self.stage = TrainingStage.finetune.value # set staging before re-instantiating model, as staging determines how model is initialized
        self.model = self._instantiate_model(model_path=pretrained_model_path)

    @property
    def is_pretrain_stage(self) -> bool:
        return (self.stage == TrainingStage.pretrain.value)

    def pretrain_forward(self, X: torch.Tensor) -> Tuple[str, torch.Tensor, torch.Tensor]:
        cluster = (self.input_fdim != self.fshape) # Same as original implementation in the SSAST paper/repo by (Gong et. al)
        batch_accuracy, discriminative_loss = self.model(X, 'pretrain_mpc', mask_patch=self.pretext_masked_patches, cluster=cluster)
        generative_loss = self.model(X, 'pretrain_mpg', mask_patch=self.pretext_masked_patches, cluster=cluster)

        discriminative_loss = discriminative_loss.mean() # InfoNCE loss (Info Noise-Contrasting Estimation) 
        generative_loss = generative_loss.mean() # MSE loss

        loss = discriminative_loss + self.generative_loss_weight * generative_loss
        return loss, batch_accuracy
        
    def reshape(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        X, Y = batch # X.shape: (batch_size, 1, nmels, time_frames)
        # self.model requires input shape (batch_size, time_frames, nmels)
        X = X.permute(0, 1, 3, 2) # X.shape: (batch_size, 1, time_frames, nmels)
        X = X.squeeze(dim=1) # X.shape: (batch_size, time_frames, nmels)
        return X, Y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Mapping[str, torch.Tensor]:
        X, Y = self.reshape(batch)
        if self.is_pretrain_stage:
            loss, batch_accuracy = self.pretrain_forward(X)
            self.log("train_pretext_loss", loss)
            return dict(loss=loss)
        else:
            Yhat = self.finetune_activation(self.model(X, task=self.clip_representation))
            loss = self.finetune_loss(Yhat, Y)
            self.log("train_loss", loss)
            return dict(loss=loss)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Mapping[str, torch.Tensor]:
        X, Y = self.reshape(batch)
        if self.is_pretrain_stage:
            loss, batch_accuracy = self.pretrain_forward(X)
            self.log("val_pretext_loss", loss)
            self.val_pretext_batch_accuracy.update(batch_accuracy=batch_accuracy)
            return dict(loss=loss)
        else:
            Yhat = self.finetune_activation(self.model(X, task=self.clip_representation))
            loss = self.finetune_loss(Yhat, Y)
            self.log("val_loss", loss)
            self.val_finetune_metrics.update(Yhat.float(), Y.int())
            self.val_confusion_matrix.update(Yhat.float(), Y.int())
            return dict(loss=loss)

    def validation_epoch_end(self, outputs: Any) -> None:
        if self.is_pretrain_stage:
            accs = self.val_pretext_batch_accuracy.compute()
            self.log("val_pretext_acc", accs)
            self.val_pretext_batch_accuracy.reset()
        else:
            metrics = self.val_finetune_metrics.compute(step="val")
            confusion = self.val_confusion_matrix.compute()
            
            self.log_metrics("val", metrics, confusion)
            
            self.val_finetune_metrics.reset()
            self.val_confusion_matrix.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Mapping[str, torch.Tensor]:
        X, Y = self.reshape(batch)
        if self.is_pretrain_stage:
            loss, batch_accuracy = self.pretrain_forward(X)
            self.log("test_pretext_loss", loss)
            self.test_pretext_batch_accuracy.update(batch_accuracy=batch_accuracy)
            return dict(loss=loss)
        else:
            Yhat = self.finetune_activation(self.model(X, task=self.clip_representation))
            loss = self.finetune_loss(Yhat, Y)
            self.log("test_loss", loss)
            self.test_finetune_metrics.update(Yhat.float(), Y.int())
            self.test_confusion_matrix.update(Yhat.float(), Y.int())
            return dict(loss=loss)

    def test_epoch_end(self, outputs: Any) -> None:
        if self.is_pretrain_stage:
            accs = self.test_pretext_batch_accuracy.compute()
            self.log("test_pretext_acc", accs)
            self.test_pretext_batch_accuracy.reset()
        else:
            metrics = self.test_finetune_metrics.compute(step="test")
            confusion = self.test_confusion_matrix.compute()
            
            self.log_metrics("test", metrics, confusion)
            
            self.test_finetune_metrics.reset()
            self.test_confusion_matrix.reset()

    def log_metrics(self, step: str, metrics: Mapping[str, torch.Tensor], confusion: torch.Tensor) -> None:
        for name, metric in metrics.items():
            self.log(name, metric)

        biophonic_confusion = confusion[0]
        anthropogenic_confusion = confusion[1]
        
        self.logger.experiment.log({f"{step}_bio_confusion_matrix": customwandbplots.confusion_matrix(self.logger, biophonic_confusion, class_names=["not bio", "bio"], title=f"{step} confusion matrix (biophonic)")})
        self.logger.experiment.log({f"{step}_anth_confusion_matrix": customwandbplots.confusion_matrix(self.logger, anthropogenic_confusion, class_names=["not anth", "anth"], title=f"{step} confusion matrix (anthropogenic)")})

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay, 
            betas=self.betas
        )
        # TODO: Implement learning rate scheduler(s)
        return dict(optimizer=optimizer)

if __name__ == "__main__":
    model = SSASTLightningWrapper(
        learning_rate = 1e-4,
        weight_decay = 5e-7,
        betas = (0.95, 0.999),
        stage = TrainingStage.pretrain.value,
        fstride = 16,
        tstride = 16,
        fshape = 16,
        tshape = 16,
        input_fdim = 1024,
        input_tdim = 128,
        n_model_outputs = 2,
        model_size = SSASTModelSize.base.value,
        pretext_masked_patches = 400,
        generative_loss_weight = 10.0,
        clip_representation = ClipRepresentationMethod.average_tokens.value,
        class_names = None
    )
