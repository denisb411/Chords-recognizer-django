# Chords Recognizer
A Django server for chord recognition from audio data trained with ML models

## Description

This repository contains models written with Python's Scikit-learn and Tensorflow libraries to deal with the chords classification task
and currently is much more a Proof of Concept rather than a ready-for-production application.

On this research I tested several models using several combinations of signal pre-processing to find out the scenarios behavior.
The dataset was recorded by myself using a Classical (Nylon) Guitar, 48 classes (minor, major, minor 7 and major 7) and with 2 
different samples size.

###About the datasets:
* ~14300 samples with 1024 features (audio samples) each, 48 classes - training with Scikit-learn library
* ~3200 samples with 20480 samples each, 48 classes - training with Tensorflow (GPU - GTX 1060 6GB)

###Some of the used models:
Scikit-learn:
* Linear Regression
* Support Vector Machine
* Decision Trees
* Logistic Regression
* Multi-layer Perceptron

Tensorflow:
* Neural Networks
* Convolutional Neural Networks

###Some of the pre-processment used:
* Signal Windowing
* Fourier Transform
* Short-time Fourier Transform
* Chroma feature extraction


You can follow all the research on the following article (Portuguese-Brazil): https://goo.gl/F614Kf
