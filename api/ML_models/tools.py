import tensorflow as tf
import librosa
import numpy as np

def leaky_relu(alpha=0.01):
	def parametrized_leaky_relu(z, name=None):
		return tf.maximum(alpha * z, z, name=name)
	return parametrized_leaky_relu

def preprocess_sample(sample):
	""" This preprocess has to be the same as the training! """

	for ii in range(len(sample)):
		if ii < 1000: #ignore frequencies greater than 2kHz
			X_fft_new[ii] = sample[ii]
	sample = np.fft.ifft(X_fft_new)

	sample = np.array(sample, dtype=np.float)
	sample = librosa.feature.chroma_stft(y=sample, sr=44100, n_fft=20480, hop_length=258)
	sample = np.atleast_3d(sample)
	sample = np.reshape(sample,(1, 12,80))

	return sample