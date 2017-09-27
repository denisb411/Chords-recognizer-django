from api.AI.classifier import Classifier
from api.AI.dataset_loader import DatasetLoader

classifier = Classifier()
classifier.train(dataset.X_train, dataset.y_train,
		  dataset.X_valid, dataset.y_valid,
		  dataset.X_test, dataset.y_test)