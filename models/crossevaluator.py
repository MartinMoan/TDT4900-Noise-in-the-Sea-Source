#!/usr/bin/env python3
import multiprocessing
from multiprocessing.pool import ThreadPool
import sys
import pathlib

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import IModelProvider, ILoggerFactory, IDatasetVerifier, ICrossEvaluator, ITracker, IDatasetProvider, ITrainer, IEvaluator, IMetricComputer, IFolder, ISaver
from tools.mapagg import MappingAggregator
import inspect

def verify(func):
    def nested(self, *args, **kwargs):
        spec = inspect.getfullargspec(func)
        names = spec.args
        named_args = {names[i]: args[i] for i in range(min(len(names), len(args)))}
        named_args = {key: value for key, value in named_args.items() if key != "self"}
        named_args = {**named_args, **kwargs}
        
        for arg, input_value in named_args.items():
            if arg in spec.annotations:
                expected_type = spec.annotations[arg]
                if not isinstance(input_value, expected_type):
                    raise TypeError(f"Argument {arg} has incorrect type. Expected {expected_type} but received {type(input_value)}")

        func(self, *args, **kwargs)
    return nested

class CrossEvaluator(ICrossEvaluator):
    @verify
    def __init__(
        self, 
        model_provider: IModelProvider, 
        logger_factory: ILoggerFactory,
        dataset_provider: IDatasetProvider,
        dataset_verifier: IDatasetVerifier,
        tracker: ITracker,
        folder: IFolder,
        trainer: ITrainer,
        evaluator: IEvaluator,
        metric_computer: IMetricComputer,
        saver: ISaver,
        n_fold_workers: int):
        
        self._logger_factory = logger_factory
        self._logger = logger_factory.create_logger()
        self._model_provider = model_provider
        self._dataset_verifier = dataset_verifier
        self._dataset_provider = dataset_provider
        self._tracker = tracker
        self._folder = folder
        self._trainer = trainer
        self._evaluator = evaluator
        self._metric_computer = metric_computer
        self._saver = saver
        self._n_fold_workers = n_fold_workers

    def afold(self, fold, dataset, training_samples, test_samples):
        bar = "----------------------------------------"
        msg = f"START FOLD {fold}"
        space = int( (len(bar) - len(msg)) / 2)
        halfbar = bar[:space]
        
        self._logger.log(bar)
        self._logger.log(f"{halfbar}{msg}{halfbar}")
        self._logger.log(bar)
        self._logger.log(f"Fold {fold} stats:")
        self._logger.log(f"Total dataset size: {len(dataset)}")
        self._logger.log(f"N training samples: {len(training_samples)} ({(len(training_samples) / len(dataset)):.02f}%)")
        self._logger.log(f"N test samples: {len(test_samples)} ({(len(test_samples) / len(dataset)):.02f}%)")
        self._logger.log(bar)
        
        model = self._model_provider.instantiate()

        model = self._trainer.train(model, training_samples, dataset)

        truth, predictions = self._evaluator.evaluate(model, test_samples, dataset)

        metrics = self._metric_computer(truth, predictions)
        self._logger.log(f"Fold {fold} metrics: ", metrics)
        
        self._tracker.track(
            {
                "fold": fold,
                **metrics
            }
        )
        model_parameters_path = self._saver.save(model, mode="fold_eval")
        self._logger.log(f"Saved model parameters for fold {fold} to:", model_parameters_path)
        self._logger.log(f"End fold {fold}")
        self._logger.log("----------------------------------------")
        return metrics

    def kfoldcv(self) -> None:
        self._logger.log(f"Requesting dataset from {self._dataset_provider.__class__.__name__}...")
        
        dataset = self._dataset_provider.provide()
        
        self._logger.log(f"Done!")
        self._logger.log(f"Verifying dataset before training, using verifier: {self._dataset_verifier.__class__.__name__}...")
        
        if not self._dataset_verifier.verify(dataset):
            raise Exception("Dataset could not be validated.")

        self._logger.log("Dataset was verified!")
        self._logger.log("Tracking dataset...")
        self._tracker.track_dataset(dataset)
        self._logger.log("Dataset tracking done!")
        self._logger.log(f"Starting to perform K-fold cross evaluation with folder properties: {self._folder.properties}")
        
        all_metrics = []

        with ThreadPool(processes=self._n_fold_workers) as pool:
            tasks = []
            
            for fold, (training_samples, test_samples) in enumerate(self._folder.split(dataset)):
                task = pool.apply_async(self.afold, (fold, dataset, training_samples, test_samples,))
                tasks.append(task)
            
            for task in tasks:
                all_metrics.append(task.get())
        
        cumulative_metrics = {}
        for metrics in all_metrics:
            cumulative_metrics = MappingAggregator.add(cumulative_metrics, metrics)
        cumulative_metrics = MappingAggregator.div(cumulative_metrics, len(all_metrics))
        self._logger.log("Mean metrics:", cumulative_metrics)
        self._tracker.track({"eval": cumulative_metrics}) 

if __name__ == "__main__":
    import torch

    from tracking.logger import Logger, LogFormatter
    from tracking.tracker import Tracker
    from models.trainer import Trainer
    from tracking.saver import Saver
    from datasets.folder import BasicKFolder
    from models.evaluator import Evaluator
    from models.optimizer import AdamaxProvider, GeneralOptimizerProvider
    from metrics import BinaryMetricComputer
    from datasets.balancing import BalancedKFolder, DatasetBalancer
    from tracking.loggerfactory import LoggerFactory

    from mocks.mockmodelprovider import MockModelProvider
    from mocks.mockdatasetprovider import MockDatasetProvider
    from mocks.mocktensordataset import MockTensorDataset
    from mocks.mockdatasetverifier import MockDatasetVerifier
    from mocks.mocktracker import MockTracker

    batch_size = 16
    epochs = 3
    lossfunction = torch.nn.BCEWithLogitsLoss()
    num_workers = multiprocessing.cpu_count()
    lr = 0.001
    weight_decay = 1e-5
    kfolds = 5

    logger_factory = LoggerFactory(
        logger_type=Logger, 
        logger_args=(LogFormatter(),)
    )
    
    model_provider = MockModelProvider(output_shape=(2,))
    
    dataset = MockTensorDataset(
        size=100, 
        label_shape=(2,), 
        feature_shape=(1, 4, 8),
        classes={"Anthropogenic": 0, "Biophonic": 1},
        logger_factory=logger_factory
    )

    dataset_provider = MockDatasetProvider(dataset)

    dataset_verifier = MockDatasetVerifier()

    tracker = MockTracker(logger_factory=logger_factory)
    
    folder = BasicKFolder(n_splits=kfolds, shuffle=True)

    optimizer_provider = GeneralOptimizerProvider(
        optimizer_type=torch.optim.Adamax,
        optimizer_args=(),
        optimizer_kwargs={"lr": lr, "weight_decay": weight_decay}
    )

    metric_computer = BinaryMetricComputer(
        logger_factory=logger_factory,
        class_dict=dataset.classes(),
        threshold=0.5
    )
    
    trainer = Trainer(
        logger_factory=logger_factory,
        optimizer_provider=optimizer_provider,
        tracker=tracker,
        metric_computer=metric_computer,
        batch_size=batch_size,
        epochs=epochs,
        lossfunction=lossfunction,
        num_workers=num_workers,
    )

    evaluator = Evaluator(logger_factory=logger_factory, batch_size=16)

    saver = Saver(logger_factory=logger_factory)

    cv = CrossEvaluator(
        model_provider=model_provider, 
        logger_factory=logger_factory,
        dataset_provider=dataset_provider,
        dataset_verifier=dataset_verifier,
        tracker=tracker,
        folder=folder,
        trainer=trainer,
        evaluator=evaluator,
        metric_computer=metric_computer,
        saver=saver,
        n_fold_workers=min(kfolds, multiprocessing.cpu_count())
    )

    cv.kfoldcv()
