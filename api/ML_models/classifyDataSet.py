from api.ML_models.loadDataSet import loadMainData, loadCachedData, loadTestData, loadPreprocessedSamples, loadPreprocessedSamples_chroma
from api.ML_models.tensorflow_models import DNNClassifier, CNNClassifier


import numpy as np
import pandas as pd
from collections import Counter

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn import utils

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from django.core.exceptions import SuspiciousOperation

from sklearn.metrics import accuracy_score

import librosa

import tensorflow as tf

import os

global trained
trained = False

def get_model_params():
	gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
	gvar_names = list(model_params.keys())
	assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign") for gvar_name in gvar_names}
	init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
	feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
	tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

def leaky_relu(alpha=0.01):
		def parametrized_leaky_relu(z, name=None):
			return tf.maximum(alpha * z, z, name=name)
		return parametrized_leaky_relu


def fitAndPredict (name, model, trainX, trainY):
	k = 3
	#kfold = model_selection.KFold(n_splits=10, random_state=seed)
	scores = cross_val_score(model, trainX, trainY, cv = k, n_jobs = 4) #Evaluate a score by cross-validation
	hitRate = np.mean(scores) #calculates the average

	msg = "{0}'s hit rate: {1}".format(name, hitRate)
	print (msg)
	return hitRate, msg

def trainDNN():

	global training
	training = True

	X_train, y_train, X_valid, y_valid, X_test, y_test = loadPreprocessedSamples()

	

	global dnn
	dnn = DNNClassifier(batch_size=50, learning_rate=0.01, n_hidden_layers=5, n_neurons=300, optimizer_class=tf.train.AdagradOptimizer, activation=leaky_relu(alpha=0.01))

	# final_model_path = 'api/ML_models/final_model/'
	# final_model_file = final_model_path + 'final-model.ckpt'
	# final_model_file_any = final_model_file + '*'
	# from sklearn.metrics import accuracy_score

	# import os
	# if os.path.exists(final_model_path):
	# 	print("exist file")
	# 	#dnn.restore(final_model_file, X_train, y_train)
	# 	y_pred = dnn.predict(X_test)
	# 	score = accuracy_score(y_test, y_pred)
	# 	returnMessage = {'DNN_model': score}
	# else:
	# 	print("dont exist file")
	# 	os.makedirs(final_model_path)
	# 	#hitRate, returnMessage = fitAndPredict('DNN tensorflow', dnn, X, y)
	# 	dnn.fit(X=X_train, y=y_train, X_valid=X_valid, y_valid=y_valid)
	# 	#dnn.save(final_model_file)
	# 	y_pred = dnn.predict(X_test)
	# 	score = accuracy_score(y_test, y_pred)
	# 	returnMessage = {'DNN_model': score}

	dnn.fit(X=X_train, y=y_train, X_valid=X_valid, y_valid=y_valid)
	#dnn.save(final_model_file)
	y_pred = dnn.predict(X_test)
	score = accuracy_score(y_test, y_pred)
	returnMessage = {'DNN_model': score}

	global trained
	trained = True
	training = False

	return returnMessage

def trainCNN():

	global training
	training = True

	X_train, y_train, X_valid, y_valid, X_test, y_test = loadPreprocessedSamples_chroma()

	global cnn
	cnn = CNNClassifier(batch_size=50, learning_rate=0.01, n_hidden_layers=5, n_neurons=300, optimizer_class=tf.train.AdagradOptimizer, activation=leaky_relu(alpha=0.01))


	final_model_path = 'api/ML_models/final_model/'
	final_model_file = final_model_path + 'final-model-CNN.ckpt'
	final_model_file_any = final_model_file + '*'

	if os.path.exists(final_model_path):
		print("exist file")
		cnn.restore(final_model_file, X_train, y_train)
	else:
		print("dont exist file")
		os.makedirs(final_model_path)
		cnn.fit(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
		cnn.save(final_model_file)

	score = cnn.accuracy_score(X_test, y_test)
	returnMessage = {'DNN_model': score}

	global trained
	trained = True
	training = False

	return returnMessage

def predict(samples):
	global trained
	global training
	if trained == False:
		trainCNN()
		raise SuspiciousOperation('training') 
	if training == True:
		raise SuspiciousOperation('still training')

	global cnn
	predictedY = cnn.predict(samples)
	print(predictedY)

	return predictedY


def estimateHitRate():
	global trained
	if trained == False:
		train()

	testX, testY = loadCachedData()

	testX = np.array(testX, dtype=float)
	testY = np.array(testY, dtype=float)

	global winner
	hitRate = fitAndPredict("Winner", winner, testX, testY)
	print (hitRate)

	return hitRate


def testModels():
	if trained == False:
		train()

	testX, testY = loadCachedData()

	testX = np.array(testX, dtype=float)
	testY = np.array(testY, dtype=float)

	results = {}
	returnMessage = {}

	results, returnMessage = classifyModels(testX, testY)

	print (returnMessage)
	
	return returnMessage

def leaky_relu(alpha=0.01):
	def parametrized_leaky_relu(z, name=None):
		return tf.maximum(alpha * z, z, name=name)
	return parametrized_leaky_relu

def classifyModels(X, Y, specificModel, X_valid=None, y_valid=None):

	results = {}
	returnMessage = {}

	if specificModel == "MLPClassifier":
		print('MLPClassifier')
		global modelMLPClassifier

		x_len = len(X[0])

		xMean = (x_len + 60) / 2

		# last: mean: 0.09261, std: 0.01624, params: {'solver': 'sgd', 'activation': 'tanh', 'hidden_layer_sizes': (170, 170, 170), 'alpha': 0.001, 'learning_rate': 'invscaling', 'learning_rate_init': 0.001}
		#modelMLPClassifier = MLPClassifier(solver = 'adam', activation = 'tanh', hidden_layer_sizes = (x_len/2, x_len/2), alpha = 0.001, learning_rate = 'invscaling', learning_rate_init = 0.001)
		modelMLPClassifier = MLPClassifier(solver = 'adam', activation = 'tanh', hidden_layer_sizes = (700, 700, 700), learning_rate = 'constant', learning_rate_init = 0.05)
		resultMLPClassifier, message = fitAndPredict("MLPClassifier", modelMLPClassifier, X, Y)
		results[resultMLPClassifier] = modelMLPClassifier
		returnMessage["MLPClassifier"] = {"message": message}

		return results, returnMessage

	if specificModel == "MLPClassifier_RandomizedSearchCV":

		x_len = len(X[0])

		print('MLPClassifier_RandomizedSearchCV')
		modelMLPClassifier = MLPClassifier()

		#test1:
		params = {"activation": ['identity', 'logistic', 'tanh', 'relu'], "solver": ['lbfgs', 'sgd', 'adam'], "learning_rate": ['constant', 'invscaling', 'adaptive'], "hidden_layer_sizes": [(x_len,),(x_len,x_len),(x_len,x_len,x_len)], "learning_rate_init": [0.001, 0.01, 0.1], "alpha": [0.0001, 0.001, 0.01]}
		#params = {"hidden_layer_sizes": [(1024,),(1024,1024),(1024,1024,1024)]}


		from sklearn.grid_search import RandomizedSearchCV

		randomSearch = RandomizedSearchCV(modelMLPClassifier, params, n_jobs = 4)

		randomSearch.fit(X, Y)

		print (randomSearch.grid_scores_)
		print (randomSearch.best_params_)

		returnMessage['scores'] = randomSearch.grid_scores_

		return 0, randomSearch.grid_scores_



	if specificModel == "MLPClassifier_GridSearchCV":
		from sklearn.grid_search import GridSearchCV

		print('MLPClassifier_GridSearchCV')

		x_len = len(X[0])

		xMean = (x_len + 60) / 2

		#test1:
		#params = {"hidden_layer_sizes": [(x_len,),(x_len,x_len),(x_len,x_len,x_len)], "learning_rate_init": [0.001, 0.01], "alpha": [0.0001, 0.001]}
		#params = {"activation": ['identity', 'logistic', 'tanh', 'relu'], "solver": ['lbfgs', 'sgd', 'adam'], "learning_rate": ['constant', 'invscaling', 'adaptive'], "hidden_layer_sizes": [(x_len,),(x_len,x_len),(x_len,x_len,x_len)], "learning_rate_init": [0.001, 0.01, 0.1], "alpha": [0.0001, 0.001, 0.01]}


		#test2:
		#params = {"activation": ['identity', 'tanh'], "solver": ['sgd', 'adam'], "learning_rate": ['invscaling', 'adaptive'], "hidden_layer_sizes": [(x_len,),(x_len/2,x_len/2),(x_len/3,x_len/3,x_len/3), (x_len/2,x_len/6,x_len/7,x_len/8)], "learning_rate_init": [0.001], "alpha": [0.001]}

		#test3:
		#params = {"activation": ['identity', 'tanh', 'relu'], "solver": ['adam'], "learning_rate": ['invscaling', 'adaptive', 'constant'], "hidden_layer_sizes": [(x_len,),(x_len/2,x_len/2),(x_len/3,x_len/3,x_len/3), (x_len/2,x_len/6,x_len/7,x_len/8), (x_len/2,x_len/3,x_len/4,x_len/5, x_len/6) ], "learning_rate_init": [0.001, 0.01], "alpha": [0.001, 0.01]}

		#test4:
		#params = {"activation": ['identity', 'tanh', 'relu'], "solver": ['adam'], "learning_rate": ['invscaling', 'adaptive', 'constant'], "hidden_layer_sizes": [(xMean,),(xMean/2,xMean/2)], "learning_rate_init": [0.001, 0.01, 0.1], "alpha": [0.0001, 0.001, 0.01, 0.1]}

		#test5:
		#params = {"activation": ['tanh'], "solver": ['adam'], "learning_rate": ['invscaling', 'adaptive', 'constant'], "hidden_layer_sizes": [(xMean,),(xMean/2,xMean/2), (xMean/2,xMean/2,xMean/2), (xMean/3,xMean/3,xMean/3)], "learning_rate_init": [0.01], "alpha": [0.01, 0.1, 1, 10, 100]}

		#test6:
		#params = {"activation": ['tanh', 'logistic', 'relu'], "solver": ['adam'], "learning_rate": ['invscaling', 'adaptive', 'constant'], "hidden_layer_sizes": [(x_len,), (x_len/2), (x_len/2)], "learning_rate_init": [0.01, 0.05], "alpha": [0.1, 0.5]}

		#test7:
		params = {"activation": ['relu'], "solver": ['adam'], "learning_rate": ['invscaling', 'adaptive', 'constant'], "hidden_layer_sizes": [(xMean,), (xMean/2, xMean/2)], "learning_rate_init": [0.0001], "alpha": [0.1, 0.01]}


		#gs = GridSearchCV(MLPClassifier(), param_grid={'learning_rate': [0.05, 0.01, 0.005, 0.001], 'hidden0__units': [4, 8, 12], 'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})

		gs = GridSearchCV(MLPClassifier(), param_grid= params, n_jobs = 4)

		#gs = GridSearchCV(MLPClassifier(), param_grid=parameters)

		gs.fit(X, Y)	

		print (gs.grid_scores_)

		returnMessage['scores'] = gs.grid_scores_

		return 0, gs.grid_scores_

	if specificModel == "LogisticRegression":

		print('LogisticRegression')

		x_len = len(X[0])

		#test1:
		params = {'C': [1.0, 0.1], 'fit_intercept': [True, False], 'solver': ['newton-cg', 'liblinear', 'sag']}


		#gs = GridSearchCV(MLPClassifier(), param_grid={'learning_rate': [0.05, 0.01, 0.005, 0.001], 'hidden0__units': [4, 8, 12], 'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})

		from sklearn.grid_search import GridSearchCV
		gs = GridSearchCV(LogisticRegression(), param_grid= params, n_jobs = 4)

		#gs = GridSearchCV(MLPClassifier(), param_grid=parameters)

		gs.fit(X, Y)	

		print (gs.grid_scores_)

		returnMessage['scores'] = gs.grid_scores_

		return 0, gs.grid_scores_

	if specificModel == "DNNClassifier_RandomizedSearchCV":

		print('DNNClassifier_RandomizedSearchCV')

		from sklearn.model_selection import RandomizedSearchCV

		X_train, y_train, X_valid, y_valid, X_test, y_test = loadPreprocessedSamples()

		""" best for this test:

		param_distribs = {
			"n_neurons": [300, 400, 500, 700, 1000, 1500],
			"batch_size": [10, 50, 100, 500, 1000],
			"learning_rate": [0.01, 0.02, 0.05, 0.1],
			"activation": [tf.nn.relu, tf.nn.elu, leaky_relu(alpha=0.01), leaky_relu(alpha=0.1)],
			# you could also try exploring different numbers of hidden layers, different optimizers, etc.
			"n_hidden_layers": [0, 1, 2, 3, 4],
			"optimizer_class": [tf.train.AdamOptimizer, tf.train.GradientDescentOptimizer, 
								tf.train.RMSPropOptimizer, tf.train.AdagradOptimizer],
		}

		
		{'batch_size': 10, 'learning_rate': 0.05, 'n_neurons': 500, 'optimizer_class': <class 'tensorflow.python.training.adagrad.AdagradOptimizer'>, 'n_hidden_layers': 0, 'activation': <function leaky_relu.<locals>.parametrized_leaky_relu at 0x00000222255F2E18>}
		0.889

		"""

		param_distribs = {
			"n_neurons": [300, 400, 500, 700, 1000, 1500],
			"batch_size": [10, 50, 100],
			"learning_rate": [0.01, 0.05, 0.1],
			"activation": [leaky_relu(alpha=0.01), leaky_relu(alpha=0.1)],
			# you could also try exploring different numbers of hidden layers, different optimizers, etc.
			"n_hidden_layers": [0, 1, 2],
			"optimizer_class": [tf.train.AdamOptimizer, 
								tf.train.AdagradOptimizer],
			"dropout_rate": [0.9, 0.5, 0.1, None]
		}

		rnd_search = RandomizedSearchCV(DNNClassifier(random_state=42), param_distribs, n_iter=50,
										fit_params={"X_valid": X_valid, "y_valid": y_valid},
										random_state=42, verbose=2)

		rnd_search.fit(X_train, y_train)

		print(rnd_search.best_params_)

		y_pred = rnd_search.predict(X_test)

		from sklearn.metrics import accuracy_score
		print(accuracy_score(y_test, y_pred))

		return 0, rnd_search.best_params_


	
	modelLogisticRegression =  LogisticRegression()
	LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
	resultsLogisticRegression, message = fitAndPredict("LogisticRegression", modelLogisticRegression, X, Y)
	results[resultsLogisticRegression] = modelLogisticRegression
	returnMessage["LogisticRegression"] = {"message": message}
	
	modelOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
	resultsOneVsRest, message = fitAndPredict("OneVsRest", modelOneVsRest, X, Y)
	results[resultsOneVsRest] = modelOneVsRest
	returnMessage["OneVsRest"] = {"message": message}
	
	modelOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
	resultOneVsOne, message = fitAndPredict("OneVsOne", modelOneVsOne, X, Y)
	results[resultOneVsOne] = modelOneVsOne
	returnMessage["OneVsOne"] = {"message": message}
	
	modelMultinomial = MultinomialNB()
	resultMultinomial, message = fitAndPredict("MultinomialNB", modelMultinomial, X, Y)
	results[resultMultinomial] = modelMultinomial
	returnMessage["MultinomialNB"] = {"message": message}

	modelDecisionTreeClassifier = DecisionTreeClassifier()
	resultDecisionTreeClassifier, message = fitAndPredict("DecisionTreeClassifier", modelDecisionTreeClassifier, X, Y)
	results[resultDecisionTreeClassifier] = modelDecisionTreeClassifier
	returnMessage["DecisionTreeClassifier"] = {"message": message}

	modelKNeighborsClassifier = KNeighborsClassifier()
	resultKNeighborsClassifier, message = fitAndPredict("KNeighborsClassifier", modelKNeighborsClassifier, X, Y)
	results[resultKNeighborsClassifier] = modelKNeighborsClassifier
	returnMessage["KNeighborsClassifier"] = {"message": message}

	modelLinearDiscriminantAnalysis = LinearDiscriminantAnalysis()
	resultLinearDiscriminantAnalysis, message = fitAndPredict("LinearDiscriminantAnalysis", modelLinearDiscriminantAnalysis, X, Y)
	results[resultLinearDiscriminantAnalysis] = modelLinearDiscriminantAnalysis
	returnMessage["LinearDiscriminantAnalysis"] = {"message": message}

	modelGaussianNB = GaussianNB()
	resultGaussianNB, message = fitAndPredict("GaussianNB", modelGaussianNB, X, Y)
	results[resultGaussianNB] = modelGaussianNB
	returnMessage["GaussianNB"] = {"message": message}

	modelMLPClassifier = MLPClassifier()
	resultMLPClassifier, message = fitAndPredict("MLPClassifier", modelMLPClassifier, X, Y)
	results[resultMLPClassifier] = modelMLPClassifier
	returnMessage["MLPClassifier"] = {"message": message}

	modelSGDClassifier = SGDClassifier()
	resultSGDClassifier, message = fitAndPredict("SGDClassifier", modelSGDClassifier, X, Y)
	results[resultSGDClassifier] = modelSGDClassifier
	returnMessage["SGDClassifier"] = {"message": message}

	modelSVC = SVC()
	resultSVC, message = fitAndPredict("SVC", modelSVC, X, Y)
	results[resultSVC] = modelSVC
	returnMessage["SVC"] = {"message": message}
	
	return results, returnMessage

def testData(dataPercentage,bufferSize,testCase, trainWithDecibels, dataset, applyFFT, applyWindowFunction, windowFunction, mergeDatasets, secondDataset, specificModel):

	X,Y = loadTestData(dataPercentage,bufferSize,testCase, trainWithDecibels, dataset, applyFFT, applyWindowFunction, windowFunction, mergeDatasets, secondDataset)

	#X, Y = loadPreprocessedSamples()

	X = np.array(X, dtype=float)
	Y = np.array(Y, dtype=float)
	
	results, returnMessage = classifyModels(X, Y, specificModel)

	print (returnMessage)
	
	return returnMessage

#trainCNN()