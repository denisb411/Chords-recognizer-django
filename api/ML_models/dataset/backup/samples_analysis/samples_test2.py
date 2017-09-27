import csv
import pandas as pd
from django.db import DataError
import numpy as np
from django.core.exceptions import SuspiciousOperation

def loadTestData(instrumentType,dataPercentage,bufferSize,testCase, trainWithDecibels):
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
				1 -> means that will test 10 samples for each chord between minor/major chords
				2 -> means that will test 50 samples for each chord between minor/major chords
				3 -> means that will test 10 samples for each chord between major/minor/maj7/m7 chords
				4 -> means that will test 50 samples for each chord between major/minor/maj7/m7 chords

	"""
	print testCase
	print instrumentType
	if testCase == 1:
		samplesNumber = 10
		print 'ola'
		if instrumentType == "steelGuitar":
			df = pd.read_csv('samples_steelGuitar_8192_test.csv') #this file contains samples of all major/minor chords
		elif instrumentType == "nylonGuitar":
			df = pd.read_csv('api//AI//dataset//backup//nylonDataset.csv')
		else:
			raise SuspiciousOperation('invalid instrument')
	elif testCase == 2:
		samplesNumber = 50
		if instrumentType == "steelGuitar":
			df = pd.read_csv('api//AI//dataset//backup//steelGuitar_8192_r02.csv')
		elif instrumentType == "nylonGuitar":
			df = pd.read_csv('api//AI//dataset//backup//nylonDataset.csv')
		else:
			raise SuspiciousOperation('invalid instrument')
	elif testCase == 3:
		samplesNumber = 10
		if instrumentType == "steelGuitar":
			df = pd.read_csv('api//AI//dataset//backup//teststeelGuitar.csv') #this file contains samples of all major/minor/maj7/m7 chords
		elif instrumentType == "nylonGuitar":
			df = pd.read_csv('api//AI//dataset//backup//nylonDataset.csv')
		else:
			raise SuspiciousOperation('invalid instrument')
	elif testCase == 4:
		samplesNumber = 50
		if instrumentType == "steelGuitar":
			df = pd.read_csv('api//AI//dataset//backup//teststeelGuitar.csv')
		elif instrumentType == "nylonGuitar":
			df = pd.read_csv('api//AI//dataset//backup//nylonDataset.csv')
		else:
			raise SuspiciousOperation('invalid instrument')
	else:
		raise SuspiciousOperation('invalid test case')

	originalBufferSize = (8192)
	
	if bufferSize > originalBufferSize:
		raise SuspiciousOperation('enter a bufferSize <= 8192')

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

	resolution = round(originalBufferSize/bufferSize)
	for i in range(len(df)):
		a = df.iloc[i,:-1]
		newDfFloat = a.reshape(-1,resolution).mean(axis=1)
		newDfFloat = fftData=np.abs(np.fft.rfft(newDfFloat)) #Fourier transform on each sample
		poweredDf = []
		if trainWithDecibels == 1:
			for j in range(len(newDfFloat)):
				powered = newDfFloat[j] * pow(10,20)
				poweredDf.append(round(20 * np.log10(powered)))
			newDfFloat = np.array(poweredDf, dtype=float)
		newDfFloat = np.append(newDfFloat, df.iloc[i,-1])
		try:
			newDf = np.vstack([newDf,newDfFloat])
		except:
			newDf = np.array(newDfFloat,dtype=float)

	np.random.shuffle(newDf)
	if dataPercentage > 1:
		return HttpResponseBadRequest('enter a dataPercentage <= 1')
	rangeOfColumns = int(dataPercentage*(bufferSize/2))
	
	count = 0
	filteredX = []
	filteredY = []
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

	return filteredX, filteredY