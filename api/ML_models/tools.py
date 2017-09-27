import tensorflow as tf

def leaky_relu(alpha=0.01):
	def parametrized_leaky_relu(z, name=None):
		return tf.maximum(alpha * z, z, name=name)
	return parametrized_leaky_relu