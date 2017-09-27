import csv
import pandas as pd
from django.db import DataError
import numpy as np
from django.core.exceptions import SuspiciousOperation
#import api.ML_models.audioFeatureExtraction as aF
#import api.ML_models.cqt as cqt

import librosa


def loadMainData():
	try:
		df = pd.read_csv('api//ML_models//dataset//MainData.csv')
		X = df.iloc[:,0:-1]
		Y = df.iloc[:,-1]
	except:
		df = pd.read_csv('api//ML_models//dataset//bufferData.csv')


	for i in range(len(df)):
		a = df.iloc[i,:-1]
		newDfFloat = fftData=np.abs(np.fft.rfft(a)) #Fourier transform on each sample
		newDfFloat = np.append(newDfFloat, df.iloc[i,-1])
		try:
			newDf = np.vstack([newDf,newDfFloat])
		except:
			newDf = np.array(newDfFloat,dtype=float)

	X = newDf[:,0:-1]
	Y = newDf[:,-1]

	return X, Y

def loadPreprocessedSamples():

	df = pd.read_csv('api//ML_models//dataset//backup//samples_nylonGuitar_20480_fingerNpick_Mm7_R1A.csv')
	# df = pd.read_csv('samples_nylonGuitar_1024_Mm7_R03.csv')
	# df = pd.read_csv('../CachedData.csv')

	X_load = np.array(df.iloc[:,:-1], dtype=np.float)
	y_load = np.array(df.iloc[:,-1], dtype=np.float)

	import os

	processedData_path = "api//ML_models//dataset//backup//preprocessedSamples.data"

	processedX = np.zeros((len(X_load),10240), dtype=np.float)
	processedy = np.zeros(len(y_load), dtype=np.float)

	for i in range(len(X_load)):
		sample = np.array(X_load[i], dtype=np.float)
		sample = sample*np.hamming(20480)
		sample = np.abs(np.fft.rfft(sample))[1:]
	#     sample = np.reshape(sample,(256,255, 1))
	#     sample = np.append(sample, y[i])
		processedX[i] = sample
		processedy[i] = y_load[i]
		if i % 1000 == 0:
			print(i)
			print(sample.shape)

	from sklearn.utils import shuffle
	print(processedy)
	sprocessedX, sprocessedy = shuffle(processedX, processedy)
	print(len(sprocessedX))


	# for i in range(len(sprocessedy)):
	# 	sprocessedy[i] = (sprocessedy[i]) - 1
		
	trainRange = int(len(sprocessedX) * 0.8)
	validRange = int(len(sprocessedX) * 9)
	testRange = int(len(sprocessedX) * 0.1)


	X_train = np.array(sprocessedX[:trainRange], dtype=np.float)
	y_train = np.array(sprocessedy[:trainRange], dtype=np.float)

	X_valid = np.array(sprocessedX[trainRange:validRange], dtype=np.float)
	y_valid = np.array(sprocessedy[trainRange:validRange], dtype=np.float)

	X_test = np.array(sprocessedX[testRange:], dtype=np.float)
	y_test = np.array(sprocessedy[testRange:], dtype=np.float)
	print(y_test[1])

	#print(X_train.shape,y_train.shape, X_valid.shape, y_valid.shape)

	len(sprocessedX)

	return X_train, y_train, X_valid, y_valid, X_test, y_test

def loadPreprocessedSamples_chroma():

	import os

	processedDataX_path = "preprocessedSamples_X_samples_allGuitar_20480_Mm7_R1A.data"
	processedDatay_path = "preprocessedSamples_y_samples_allGuitar_20480_Mm7_R1A.data"
	processedData_path = ""

	if os.path.isfile(processedDataX_path): #if already preprocessed
		processedX = np.load(processedDataX_path)
		processedy = np.load(processedDatay_path)
	else:
		df = pd.read_csv('api//ML_models//dataset//backup//samples_allGuitar_20480_Mm7_R1A.csv')
		# df = pd.read_csv('samples_nylonGuitar_1024_Mm7_R03.csv')
		# df = pd.read_csv('../CachedData.csv')

		X_load = np.array(df.iloc[:,:-1], dtype=np.float)
		y_load = np.array(df.iloc[:,-1], dtype=np.float)
		processedX = np.zeros((len(X_load),12,80,1), dtype=np.float)
		processedy = np.zeros(len(y_load), dtype=np.float)
		for i in range(len(X_load)):
			# sample = librosa.core.stft(y=X_load[i], n_fft=2048, win_length=128, window='hamming', center=True, dtype=np.float32, pad_mode='reflect')
			sample = librosa.feature.chroma_stft(y=X_load[i], sr=44100, n_fft=20480, hop_length=258)
			sample = np.atleast_3d(sample)
			processedX[i] = sample
			processedy[i] = y_load[i]
		
		processedX.dump(processedDataX_path)
		processedy.dump(processedDatay_path)

	from sklearn.utils import shuffle
	print(processedy)
	sprocessedX, sprocessedy = shuffle(processedX, processedy)
	print(len(sprocessedX))
		
	trainRange = int(len(sprocessedX) * 0.9)
	validRange = int(len(sprocessedX) * 0.91)
	testRange = int(len(sprocessedX) * 0.1)


	X_train = np.array(sprocessedX[:trainRange], dtype=np.float)
	y_train = np.array(sprocessedy[:trainRange], dtype=np.float)

	X_valid = np.array(sprocessedX[trainRange:validRange], dtype=np.float)
	y_valid = np.array(sprocessedy[trainRange:validRange], dtype=np.float)

	X_test = np.array(sprocessedX[testRange:], dtype=np.float)
	y_test = np.array(sprocessedy[testRange:], dtype=np.float)
	print(y_test[1])

	#print(X_train.shape,y_train.shape, X_valid.shape, y_valid.shape)

	len(sprocessedX)


	return X_train, y_train, X_valid, y_valid, X_test, y_test 


def loadTestData(dataPercentage,bufferSize,testCase, trainWithDecibels, dataset, applyFFT, applyWindowFunction, windowFunction, mergeDatasets, secondDataset):
	""" expected inputs:
			instrumentType:
				"steelGuitar"
				"nylonGuitar" //TODO
			dataPercentage:
				0 < dataPercentage <= 1
			bufferSize:
				8192
				4096
				2048
				1024....
			testCase:
				1 -> means that will test 10 samples for each chord of the selected dataset
				2 -> means that will test 50 samples for each chord of the selected dataset
				3 -> means that will test 100 samples for each chord of the selected dataset
				4 -> means that will test all samples for each chord of the selected dataset

	"""
	print (testCase)
	df = pd.read_csv('api//ML_models//dataset//backup//%s' % dataset)
	#if mergeDatasets == 1:
	#	df2 = pd.read_csv('api//ML_models//dataset//backup//%s' % secondDataset)
	#	df = df.append(df2)

	

	if testCase == 1:
		samplesNumber = 10
	elif testCase == 2:
		samplesNumber = 50
	elif testCase == 3:
		samplesNumber = 100
	elif testCase == 4:
		samplesNumber = 9999
	else:
		raise SuspiciousOperation('invalid test case')

	originalBufferSize = (1024)

	filteredX = []
	filteredY = []

	if trainWithDecibels == 1:
		return 0, 0

	#print applyFFT

	#if applyFFT == 0:
	#	for i in range(len(df)):
	#		sample = df.iloc[i,:]
	#		filteredX.append(sample[0:-1])
	#		filteredY.append(sample[-1])

	#	print('not applying fft')

	#	return filteredX, filteredY

	if dataPercentage > 1:
		return HttpResponseBadRequest('enter a dataPercentage <= 1')
	
	if bufferSize > originalBufferSize:
		raise SuspiciousOperation('enter a bufferSize <= 1024')

	# resolution = round(originalBufferSize/bufferSize)
	# dfLenght = originalBufferSize/resolution + 1
	# for i in range(len(df)):
	# 	a = df.iloc[i,:-1]
	# 	newDfFloat = a.reshape(-1,resolution).mean(axis=1)
	# 	if trainWithDecibels == 1:
	# 		newDfFloat = 20 * np.log10(newDfFloat)
	# 		print newDfFloat
	# 	newDfFloat = np.append(newDfFloat, df.iloc[i,-1])
	# 	try:
	# 		newDf = np.vstack([newDf,newDfFloat])
	# 	except:
	# 		newDf = np.array(newDfFloat,dtype=float)

	#drd = cqt.CQT(40, 22050, 12, 44100)

	resolution = round(originalBufferSize/bufferSize)
	for i in range(len(df)):
		a = np.array(df.iloc[i,:-1], dtype = np.float)
		#a = df.iloc[i,:-1]
		#newDfFloat = a.reshape(-1,resolution).mean(axis=1)
		newDfFloat = a[:bufferSize]
		if applyWindowFunction == 1:
			if windowFunction == "blackman":
				window = np.blackman(bufferSize)
				newDfFloat = newDfFloat*window

			if windowFunction == "hanning":
					window = np.hanning(bufferSize)
					newDfFloat = newDfFloat*window

			if windowFunction == "hamming":
				window = np.hamming(bufferSize)
				newDfFloat = newDfFloat*window

		#newDfFloat, TimeAxis, FreqAxis = aF.stChromagram(newDfFloat, 44100, 1024, 1024, False)
		#newDfFloat = np.array(newDfFloat, dtype = np.float)
		#newDfFloat = newDfFloat[0]

		#newDfFloat = drd.fast(newDfFloat)

		newDfFloat = np.abs(np.fft.rfft(newDfFloat)) #Fourier transform on each sample
		poweredDf = []
		if trainWithDecibels == 1:
			for j in range(len(newDfFloat)):
				powered = newDfFloat[j] * pow(10,16)
				poweredDf.append(20 * np.log10(powered))
			newDfFloat = np.array(poweredDf, dtype=np.float)

		newDfFloat = np.append(newDfFloat[1:], df.iloc[i,-1])
		try:
			newDf = np.vstack([newDf,newDfFloat])
		except:
			newDf = np.array(newDfFloat,dtype=float)

	print (newDf[0])

	if mergeDatasets == 1:
		df2 = pd.read_csv('api//ML_models//dataset//backup//%s' % secondDataset)
		for i in range(len(df2)):
			a = np.array(df2.iloc[i,:-1], dtype = np.float)
			#a = df.iloc[i,:-1]
			#newDfFloat = a.reshape(-1,resolution).mean(axis=1)
			newDfFloat = a[:bufferSize]
			if applyWindowFunction == 1:
				if windowFunction == "blackman":
					window = np.blackman(bufferSize)
					newDfFloat = newDfFloat*window

				if windowFunction == "hanning":
					window = np.hanning(bufferSize)
					newDfFloat = newDfFloat*window

				if windowFunction == "hamming":
					window = np.hamming(bufferSize)
					newDfFloat = newDfFloat*window

			#newDfFloat = np.abs(np.fft.rfft(newDfFloat)) #Fourier transform on each sample
			poweredDf = []
			if trainWithDecibels == 1:
				for j in range(len(newDfFloat)):
					powered = newDfFloat[j]# * pow(10,16)
					poweredDf.append(20 * np.log10(powered))
				newDfFloat = np.array(poweredDf, dtype=np.float)
			newDfFloat = np.append(newDfFloat[1:], df2.iloc[i,-1])
			newDf = np.vstack([newDf,newDfFloat])

	np.random.shuffle(newDf)
	
	rangeOfColumns = int(dataPercentage*(bufferSize/2))
	
	count = 0
	
	# if testCase == 4:
	# 	for i in range(len(newDf)):
	# 		sample = newDf[i,:]
	# 		filteredX.append(sample[0:rangeOfColumns])
	# 		filteredY.append(sample[-1])
		
	# 	return filteredX, filteredY

	if testCase == 4:
		newDfPd = pd.DataFrame(newDf)
		filteredX = newDfPd.iloc[:,:rangeOfColumns]
		filteredY = newDfPd.iloc[:,-1]

		print (len(filteredX))

		print ('done all samples')

		#from sklearn.decomposition import PCA

		#pca = PCA(n_components = 0.9999)
		#filteredX = pca.fit_transform(filteredX)

		#from sklearn.preprocessing import StandardScaler
		#scaler = StandardScaler()
		#filteredX = scaler.fit_transform(filteredX.astype(np.float64))


		return filteredX, filteredY

	for num in range(1,60):
		count = 0
		try:
			for i in range(len(newDf)):
				sample = newDf[i,:]
				if sample[-1] == num:
					if count < samplesNumber:
						filteredX.append(sample[0:rangeOfColumns])
						filteredY.append(sample[-1])
						count += 1
						if count == samplesNumber:
							raise StopIteration
		except StopIteration:
			pass

	print ('done filtering')

	print (len(filteredX))

	return filteredX, filteredY
		


def loadCachedData():
	df = pd.read_csv('api//ML_models//dataset//CachedData.csv')
	X = df.iloc[:,0:512]
	Y = df.iloc[:,512]

	return X, Y