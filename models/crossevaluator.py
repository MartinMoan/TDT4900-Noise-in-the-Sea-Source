#!/usr/bin/env python3
import multiprocessing
import sys
import pathlib

import git

sys.path.insert(0, str(pathlib.Path(git.Repo(pathlib.Path(__file__).parent, search_parent_directories=True).working_dir)))
from interfaces import IModelProvider, ILoggerFactory, IDatasetVerifier, ICrossEvaluator, ITracker, IDatasetProvider, ITrainer, IEvaluator, IMetricComputer, IFolder, ISaver

from tools.typechecking import verify

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
        saver: ISaver):
        
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

    def kfoldcv(self) -> None:
        self._logger.log(f"Requesting dataset from {self._dataset_provider.__class__.__name__}...")
        dataset = self._dataset_provider.provide()
        self._logger.log(f"Done!")

        self._logger.log(f"Verifying dataset before training, using verifier: {self._dataset_verifier.__class__.__name__}...")
        
        if not self._dataset_verifier.verify(dataset):
            raise Exception("Dataset could not be validated.")

        self._logger.log("Dataset was verified!")
        
        self._logger.log(f"Starting to perform K-fold cross evaluation with folder properties: {self._folder.properties}")
        for fold, (training_samples, test_samples) in enumerate(self._folder.split(dataset)):
            self._logger.log("----------------------------------------")
            self._logger.log(f"Start fold {fold}")
            
            model = self._model_provider.instantiate()

            model = self._trainer.train(model, training_samples, dataset)

            truth, predictions = self._evaluator.evaluate(model, test_samples, dataset)

            metrics = self._metric_computer(truth, predictions)
            
            properties = {
                "trainer": self._trainer.properties,
                "folder": self._folder.properties,
                "evaluator": self._evaluator.properties,
                "model_provider": self._model_provider.properties, 
                "dataset_provider": self._dataset_provider.properties
            }
            
            model_parameters_path = self._saver.save(model, mode="fold_eval")

            self._tracker.track(
                metrics,
                model.__class__.__name__, 
                model_parameters_path,
                properties,
                fold=fold)
        
            self._logger.log(f"End fold {fold}")
            self._logger.log("----------------------------------------")


if __name__ == "__main__":
    from tracking.logger import Logger
    from tracking.tracker import Tracker
    from models.trainer import Trainer
    from tracking.saver import Saver
    from datasets.folder import BasicKFolder
    from models.evaluator import Evaluator
    from metrics import BinaryMetricComputer
    from datasets.balancing import BalancedKFolder, DatasetBalancer
    from tracking.loggerfactory import LoggerFactory
    from mocks.mockmodelprovider import MockModelProvider
    from mocks.mockdatasetprovider import MockDatasetProvider
    from mocks.mocktensordataset import MockTensorDataset
    from mocks.mockdatasetverifier import MockDatasetVerifier
    from tracking.sheets import SheetClient
    import config

    logger_factory = LoggerFactory(logger_type=Logger)
    
    model_provider = MockModelProvider(output_shape=(2,))
    
    dataset = MockTensorDataset(size=1000, label_shape=(2,), feature_shape=(1, 4, 8))
    dataset_provider = MockDatasetProvider(dataset)

    dataset_verifier = MockDatasetVerifier()

    tracker = Tracker(
        logger_factory=logger_factory,
        client = SheetClient(logger_factory=logger_factory, spreadsheet_key=config.SPREADSHEET_ID, sheet_id=config.VERIFICATION_SHEET_ID)
    )
    
    folder = BasicKFolder(n_splits=5, shuffle=True)

    trainer = Trainer()

    evaluator = Evaluator(logger_factory=logger_factory, batch_size=16)

    metric_computer = BinaryMetricComputer(
        class_dict=dataset.classes(),
        threshold=0.5,
        logger=logger_factory.create_logger()
    )

    saver = Saver(logger_factory=logger_factory)

    cv = CrossEvaluator(
        model_provider, 
        logger_factory,
        dataset_provider,
        dataset_verifier,
        tracker,
        folder,
        trainer,
        evaluator,
        metric_computer,
        saver
    )

    cv.kfoldcv()
