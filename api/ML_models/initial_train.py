from .classifier import Classifier
from .dataset_loader import DatasetLoader

loader = DatasetLoader(cut_freq_above = 500)
dataset = loader.load_dataset(preprocessed_X_file='dataset/preprocessedSamples_cut_500_X_samples_allGuitar_20480_Mm7_R1D.data',
							  preprocessed_y_file='dataset/preprocessedSamples_cut_500_y_samples_allGuitar_20480_Mm7_R1D.data')

classifier = Classifier(trained_model_file='saved-model-final.ckpt')
classifier.train(dataset.X_train, dataset.y_train,
		  dataset.X_valid, dataset.y_valid,
		  dataset.X_test, dataset.y_test)