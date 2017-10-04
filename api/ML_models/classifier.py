import tensorflow as tf

import os

from .tensorflow_models import CNNClassifier, DNNClassifier
from sklearn.exceptions import NotFittedError
from .tools import leaky_relu

class Classifier(object):
	def __init__(self, model='CNN', n_hidden_layers=8, n_neurons=500, optimizer_class=tf.train.AdamOptimizer, learning_rate=0.05, 
				batch_size=600, activation=leaky_relu(), dropout_rate=0.00001,
				conv1={'conv1_fmaps': 16, 'conv1_ksize': 5, 'conv1_stride': 1, 'conv1_dropout': 0.3, 'conv1_activation': tf.nn.relu},
				conv2={'conv2_fmaps': 16, 'conv2_ksize': 5, 'conv2_stride': 1, 'conv2_dropout': 0.3, 'conv2_activation': tf.nn.relu},
				architecture=1, __trained_model_file='saved-model-final.ckpt'):
		self.__state = 'NOT_TRAINED'
		self.__trained_model_file = __trained_model_file

		if model == 'CNN':
			self.__model = CNNClassifier(n_hidden_layers=n_hidden_layers, n_neurons=n_neurons, optimizer_class=optimizer_class,
				learning_rate=learning_rate, batch_size=batch_size, activation=activation, dropout_rate=dropout_rate,
				conv1=conv1, conv2=conv2, architecture=architecture)
		elif model == 'DNN':
			self.__model = DNNClassifier(n_hidden_layers=n_hidden_layers, n_neurons=n_neurons, optimizer_class=optimizer_class,
				learning_rate=learning_rate, batch_size=batch_size, activation=activation, dropout_rate=dropout_rate)
		elif model == 'sklearn':
			self.__model = BestSKLearnModel()
		else:
			print("Invalid model name")
			raise ValueError('Invalid model name')

	def train(self, X_train, y_train, X_valid, y_valid,
				X_test, y_test):
		path = os.path.abspath(__file__)
		dir_path = os.path.dirname(path)
		models_path = dir_path + '/saved_models'
		if self.__trained_model_file == None:
			self.__advance_state()
			self.__model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)
			self.__trained_model_file = models_path + '/saved-model-final.ckpt'
			self.__model.save(self.__trained_model_file)
		else:
			self.__advance_state()
			self.__trained_model_file = models_path + '/' + self.__trained_model_file
			if os.path.isfile(self.__trained_model_file + '.meta'):
				self.__model.restore(self.__trained_model_file)
			else:
				raise ValueError('Trained model file doesnt exists')
		self.__score = self.__model.accuracy_score(X_test, y_test)	
		self.__advance_state()

	def __check_state(self):
		if self.__state == 'NOT_TRAINED':
			raise NotFittedError('Classifier not trained yet')
		elif self.__state == 'TRAINING':
			raise NotFittedError('Model is in training process')	

	def __advance_state(self):
		if self.__state == 'NOT_TRAINED':
			self.__state = 'TRAINING'
		elif self.__state == 'TRAINING':
			self.__state = 'TRAINED'

	def predict(self, X):
		self.__check_state()
		return self.__model.predict(X)

	@property
	def score(self):
		self.__check_state()
		return self.__score
