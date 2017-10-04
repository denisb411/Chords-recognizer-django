from .classifier import Classifier
from .dataset_loader import DatasetLoader

loader = DatasetLoader()
dataset = loader.load_dataset()

classifier = Classifier()
classifier.train(dataset.X_train, dataset.y_train,
		  dataset.X_valid, dataset.y_valid,
		  dataset.X_test, dataset.y_test)