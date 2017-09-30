from sklearn.utils import shuffle
import numpy as np
import os

import librosa


class DatasetLoader(object):
	def __init__(self, preprocess_type='chroma_stft', window='hamming',
				 win_length=128, n_fft=20480, hop_length=258):

		self.__preprocess_type = preprocess_type
		self.__window = window
		self.__win_length = win_length
		self.__n_fft = n_fft
		self.__hop_lenght = hop_length


	def load_dataset(self, file='main_dataset.csv', csv=False,
				train_percentage=0.8, validation_percentage=0.1,
				test_percentage=0.1,
				preprocessed_X_file='preprocessedSamples_X_samples_allGuitar_20480_Mm7_R1D.data',
				preprocessed_y_file='preprocessedSamples_y_samples_allGuitar_20480_Mm7_R1D.data'):

		path = os.path.abspath(__file__)
		dir_path = os.path.dirname(path)
		dataset_file = dir_path + '/dataset/' + file
		self.__preprocessed_X_file = dir_path + '/' + preprocessed_X_file
		self.__preprocessed_y_file = dir_path + '/' + preprocessed_y_file
		print(self.__preprocessed_X_file)
		print(self.__preprocessed_y_file)

		if os.path.isfile(self.__preprocessed_X_file) and os.path.isfile(self.__preprocessed_y_file) and csv==False:

			processed_X, processed_y = self.__load_preprocessed_data(self.__preprocessed_X_file,
																	 self.__preprocessed_y_file)
		elif os.path.isfile(dataset_file):

			try:
				data_frame = pd.read_csv(dataset_file)
			except:
				print('An exception occurred when trying to load the \
					dataset. Loading the buffer instead.')
				data_frame = pd.read_csv('api//ML_models//dataset//bufferData.csv')

			self.__state = 'PREPROCESSING'
			processed_X, processed_y = self.__preprocess_dataset(data_frame)	

		else:
			print("***********************")
			print(dataset_file)
			raise ValueError('Dataset file doesnt exist')

		self.__dataset = Dataset(processed_X, processed_y, train_percentage=train_percentage,	
						validation_percentage=validation_percentage, 
						test_percentage=test_percentage)

		self.__state = 'DONE'
		return self.__dataset

	@property
	def dataset(self):
		self.__check_state()

		return self.__dataset

	def __check_state(self):
		if self.__state == None:
			raise ValueError('Dataset not loaded!')
		elif self.__state == 'PREPROCESSING':
			raise ValueError('Dataset is still preprocessing')

	def __load_preprocessed_data(self, preprocessed_X_file, preprocessed_y_file):
		processed_X = np.load(preprocessed_X_file)
		processed_y = np.load(preprocessed_y_file)

		return processed_X, processed_y

	def __preprocess_dataset(self, data_frame):

		if self.__preprocess_type == 'fft':
			for i in range(len(data_frame)):
				row_X, row_y = data_frame.iloc[i,:-1], data_frame.iloc[i,-1]

				row_X = self.__apply_window(row_X)

				transformed_X = np.abs(np.fft.rfft(row_X)) #Fourier transform on each sample
				processedRow = np.append(transformed_X, row_y) #Append y with X
				try:
					processed_dataset = np.vstack([processed_dataset,processedRow])
				except:
					processed_dataset = np.array(processedRow,dtype=float)

			processed_X = processed_dataset[:,0:-1]
			processed_y = processed_dataset[:,-1]

			return processed_X, processed_y
		elif self.__preprocess_type == 'stft':
			X = np.array(data_frame.iloc[:,:-1], dtype=np.float)
			y = np.array(data_frame.iloc[:,-1], dtype=np.float)

			processed_X = np.zeros((len(X),12,80,1), dtype=np.float)
			processed_y = np.zeros(len(y), dtype=np.float)

			for i in range(len(X)):
				row_X = librosa.core.stft(y=X[i], n_fft=self.__n_fft, 
										 win_length=self.__win_length,
										 window=self.__window, center=True,
										 dtype=np.float32, pad_mode='reflect')

				row_X_3d = np.atleast_3d(row_X)
				processed_X[i] = row_X_3d
				processed_y[i] = y[i]
				if i % 400 == 0:
					print(i, "processed rows")

			return processed_X, processed_y
		elif self.__preprocess_type == 'chroma_stft':
			X = np.array(data_frame.iloc[:,:-1], dtype=np.float)
			y = np.array(data_frame.iloc[:,-1], dtype=np.float)

			processed_X = np.zeros((len(X),12,80,1), dtype=np.float)
			processed_y = np.zeros(len(y), dtype=np.float)

			for i in range(len(X)):
				row_X = librosa.feature.chroma_stft(y=X[i],
											sr=44100, n_fft=self.__n_fft,
											hop_length=self.__hop_lenght)

				row_X_3d = np.atleast_3d(row_X)
				processed_X[i] = row_X_3d
				processed_y[i] = y[i]
				if i % 400 == 0:
					print(i, "processed rows")

			return processed_X, processed_y



	def __apply_window(self, X):
		if self.__window == 'hamming':
			X = np.hamming(X)
		elif self.__window == 'hanning':
			X = np.hanning(X)
		elif self.__window == 'blackman':
			X = np.blackman(X)

		return X

class Dataset(object):
	def __init__(self, X, y, train_percentage=0.8,
				validation_percentage=0.1,
				test_percentage=0.1):

		shuffled_X, shuffled_y = shuffle(X, y)

		self.__define_ranges(shuffled_X, shuffled_y, train_percentage,
							validation_percentage, test_percentage)

	def __define_ranges(self, X, y, train_percentage,
						validation_percentage, test_percentage):

			trainRange = int(len(X) * train_percentage)
			validRange = int(len(X) * (validation_percentage + train_percentage))
			testRange = int(len(X) * train_percentage)

			self.__X_train = np.array(X[:trainRange], dtype=np.float)
			self.__y_train = np.array(y[:trainRange], dtype=np.float)

			self.__X_valid = np.array(X[trainRange:validRange], dtype=np.float)
			self.__y_valid = np.array(y[trainRange:validRange], dtype=np.float)

			self.__X_test = np.array(X[testRange:], dtype=np.float)
			self.__y_test = np.array(y[testRange:], dtype=np.float)

			self.__X_train_valid = np.array(X[trainRange:testRange], dtype=np.float)
			self.__y_train_valid = np.array(y[trainRange:testRange], dtype=np.float)

	@property
	def X_train(self):
		return self.__X_train
	@property
	def y_train(self):
		return self.__y_train

	@property
	def X_valid(self):
		return self.__X_valid
	@property
	def y_valid(self):
		return self.__y_valid

	@property
	def X_test(self):
		return self.__X_test
	@property
	def y_test(self):
		return self.__y_test

	@property
	def X_train_valid(self):
		return self.__X_train_valid
	@property
	def y_train_valid(self):
		return self.__y_train_valid



