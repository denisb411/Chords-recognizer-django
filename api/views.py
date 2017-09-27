from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.http import HttpRequest

from rest_framework.exceptions import ValidationError, ParseError
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser

import numpy as np
import os
import json
import csv

import api.AI.trained_model

loader = DatasetLoader()
dataset = loader.load_dataset()

@api_view(['POST'])
def clear_cached_data(request):
	if request.method == 'POST':
		print('processing clear_cached_data')
		f = open("api//ML_models//dataset//cachedData.csv", "w")
		f.truncate()
		f.close()
		return Response("OK")

	return Responde("Invalid request")

@api_view(['POST'])
def clear_main_data(request):
	if request.method == 'POST':
		sourceFile = open('api//ML_models//dataset//MainData.csv', 'r')
		data = sourceFile.read()
		f = open("api//ML_models//dataset//MainData.csv", "w")
		f.truncate()
		f.close()
		return Response("OK")

	return Response("Invalid request")

@api_view(['POST'])
def append_to_main_data (request):
	"""Used to append the dataset of cached data to the main dataset"""
	if request.method == 'POST':
		sourceFile = open('api//ML_models//dataset//cached_dataset.csv', 'r')
		data = sourceFile.read()
		with open('api//ML_models//dataset//MainData.csv', 'a') as destFile:
			destFile.write(data)
		return Response("OK")

	return Response("Invalid request")

@api_view(['POST'])
def append_to_cached_data(request):
	"""Append a sample to the cache dataset"""
	if request.method == 'POST':
		data = json.loads(request.body.decode('utf-8'))
		sample = data['sample']
		chordType = [data['chordType']]
		sample.append(data['chordType']) 
		with open('api//ML_models//dataset//CachedData.csv','a') as f:
			writer = csv.writer(f)
			writer.writerow((samples))
		return Response("OK")

	return Response("Invalid request")

@api_view(['POST'])
def predict_data(request):
	"""Predict the class for the given audio data"""
	if request.method == 'POST':
		print('Predicting data')
		data = json.loads(request.body.decode('utf-8'))
		sample = data['samples']
		preprocessed_sample = preprocess_sample(sample)
		predicted_class = int(round(trained_model.clf.predict(sample)))
		predicted_class_dic = {'predicted_class' : predicted_class}
		print(predicted_class_dic)
		return JsonResponse(predicted_class_dic)

	return Response("Invalid request")

@api_view(['GET', 'POST'])
def check_server_status(request):
	if request.method == 'GET':
		return Response("OK")
	elif request.method == 'POST':
		return Response("OK")

	return Response("invalid request")

@api_view(['POST'])
def test_hit_rate (request):
	"""Test the model's hit rate"""
	if request.method == 'POST':
		hitRate = machine.estimateHitRate()
		JSONData = JSONRenderer().render(hitRate)
		return Response(JSONData)

	return Response("Invalid request")

@api_view(['POST'])
def create_backup(request):
	"""Create a backup of the main dataset"""
	if request.method == 'POST':
		data = json.loads(request.body.decode('utf-8'))
		backup_file = data['backup_file'] #TODO: CHANGE THIS ON THE CLIENT SIDE
		if (backup_file == ""):
			raise ValidationError
		path_backup_file = ("api//ML_models//dataset//Backup//%s.csv" % backup_file)
		main_data = open('api//ML_models//dataset//MainData.csv', 'r')
		dataset = main_dataset.read()
		with open(path_backup_file, 'a') as f:
			f.write(dataset)

		#Pass a list of the backed up datasets
		files_csv = [f for f in os.listdir('api//ML_models//dataset//backup//') if f.endswith('.csv')]
		files_csv_json = json.dumps(files_csv)
		return JsonResponse(files_csv_json, safe=False)

	return Response("Invalid request")

@api_view(['POST'])
def use_backup_data_as_main_data(request):
	if request.method == 'POST':
		"""Set the informed backup dataset as the main dataset"""
		data = json.loads(request.body.decode('utf-8'))
		backup_file = data['backup_file'] #TODO: CHANGE THIS ON THE CLIENT SIDE
		if (backup_file == ""):
			raise ValidationError
		path_backup_file = ("api//ML_models//dataset//Backup//%s" % backup_file)
		backup_dataset = open(path_backup_file, 'r')
		dataset = backup_dataset.read()

		#First clear main_dataset
		f = open("api//ML_models//dataset//MainData.csv", "w")
		f.truncate()
		f.close()

		#write main_dataset with the backup dataset
		with open('api//ML_models//dataset//MainData.csv', 'a') as f:
			f.write(dataset)
		return Response("OK")

	return Response("invalid request")

@api_view(['POST'])
def test_current_trained_model (request):
	"""Test the current model performance"""
	if request.method == 'POST':
		returnMessage = machine.testCurrentTrainedModel(dataPercentage,bufferSize,testCase,trainWithDecibels,dataset)
		return JsonResponse(returnMessage)

	return Response("invalid request")

@api_view(['POST'])
def test_case(request):
	"""Used on old tests with scikit-learn"""
	if request.method == 'POST':
		data = json.loads(request.body.decode('utf-8'))
		dataPercentage = data['dataPercentage']
		bufferSize = data['bufferSize']
		testCase = data['testCase']
		trainWithDecibels = data['trainWithDecibels']
		dataset = data['dataset']
		try:
			applyFFT = data['applyFFT']
			mergeDatasets = data['mergeDatasets']
			secondDataset = data['secondDataset']
			specificModel = data['specificModel']
		except:
			applyFFT = 1
			mergeDatasets = 0
			secondDataset = ""
			specificModel = ""
		applyWindowFunction = data['applyWindowFunction']
		windowFunction = data['windowFunction']
		testsResult = machine.testData(dataPercentage,bufferSize,
								testCase,trainWithDecibels,dataset,applyFFT, 
								applyWindowFunction, windowFunction, 
								mergeDatasets, secondDataset, specificModel)
		return JsonResponse(testsResult, safe=False)

	return Response("invalid request")

@api_view(['POST'])
def list_backups (request):
	"""Resturn a list of the dataset backups"""
	if request.method == 'POST':
		files_backup = [f for f in os.listdir('api//ML_models//dataset//backup//') if f.startswith('autoBackup') == False and f.endswith('.csv') ]
		files_backup += [f for f in os.listdir('api//ML_models//dataset//backup//') if f.startswith('autoBackup') ]
		files_json = json.dumps(files_backup)
		return JsonResponse(files_backup, safe=False)

	return Response("invalid request")

def preprocess_sample(sample):
	#This preprocess has to be the same as the training!
	sample = np.array(sample, dtype=np.float)
	sample = librosa.feature.chroma_stft(y=sample, sr=44100, n_fft=20480, hop_length=258)
	sample = np.atleast_3d(sample)
	sample = np.reshape(sample,(1, 12,80))

	return sample