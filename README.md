# heartbeat_classification

Copyright (c) 2021 Timothy Morris PhD, ALL RIGHTS RESERVED

A system for classifying heartbeats based on ECG data.

Note: I am just starting the write up of this data. That can be found [HERE](write-up/ECGclassification.md).

- wavelet.py: Program to explore different wavelet decompositions on the MITBIH ECG data
- wavelet_process.py: Program to bulk process ECG data using the wavelet decomposition and save the decompositions. These files are kept in separate directories titled wavelets-X where X depends on the number of interpolation points.
- neuralnettrainer.py: Program to read wavelet decomposition data and apply a variety of parameters to an artificial neural network (MLPclassifier). Outputs a data summary of all training runs along with a model file for each training run.
- collate_results.py: Program to read summary data from neuralnettrainer.py and collate the data into a single tab separated file. Also applies each model to the test data to get more details about performance of the model. Supersedes NN_run.py
- make_graphics.py: Program to read collated data, filter it in various ways, and make various figures to represent the performance of the algorithm.
- wavelet_graphics.py: Program to produce various figures related to the wavelet decomposition of the data.
- NN_run.py Program to apply a training model to a wavelet decomposition (of the test data) to get a more detailed view of the performance of the model: NOT CURRENTLY USED
- NN.py Program to perform individual neural network training: NOT CURRENTLY USED
- verify.py: Program to apply a training model to a wavelet decomposition of data that only includes Normal vs Abnormal classifications: NOT CURRENTLY USED
