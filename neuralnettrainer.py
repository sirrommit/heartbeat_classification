#!/usr/bin/env python3
""" Train an artificial neural network to classify heartbeats from ECG
data

Copyright (c) 2021 Timothy Morris Phd
"""

import csv
import time
import pickle
import multiprocessing
from sklearn.neural_network import MLPClassifier

MAX_PROCESSES = 4

def process(parameters):
    """ Use the data to train an artificial neural network using a
    a variety of parameters.

    data is a dictionary with keys train_data, train_classification,
    test_data, and test_classification.

    parameters is the collection of parameters to be used in training.
    Each combination of parameters will be used to train an artificial
    neural network on the data. Each model will be saved along with a
    file giving the parameters used and the success for each model.
    """
    model_no=1000

    wavelets = ['mexh', 'morl', 'gaus5']

    for wavelet in wavelets:
        print("Training Neural Networks")
        data = load_preprocessed_data(wavelet)

        call_params = []
        for h_l in parameters['hidden_layer_size']:
            for act in parameters['activation']:
                for alpha in parameters['alpha']:
                    call_params.append((h_l, act, alpha, model_no, data))
                    model_no+=1

        with multiprocessing.Pool(MAX_PROCESSES) as pool:
            output = pool.starmap(single_process, call_params)

        with open("model_desc." + wavelet + ".txt", 'w') as outfile:
            for outline in output:
                outfile.write(outline)

def single_process(hidden_layers, activation, alpha, model_no, data):
    """ Use MLPClassifier to run for a single set of parameters """
    print("HL:" + str(hidden_layers) + " activation:" + str(activation) +\
          " alpha:" + str(alpha))
    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, \
                        activation=activation, alpha=alpha)
    t_start = time.perf_counter()
    clf.fit(data["train_data"], data["train_classification"])
    t_end=time.perf_counter()
    train_score = clf.score(data["train_data"], data["train_classification"])
    test_score = clf.score(data["test_data"], data["test_classification"])
    train_time = t_end - t_start
    model_file = "NN."+str(model_no)[1:]+".model"
    pickle.dump(clf,open(model_file,'wb'))
    print("For ({}, {}, {}) - train, test score: \t {:.5f} - {:.5f} \t in {:.2f} s".format(hidden_layers , activation , alpha , train_score , test_score , train_time))
    return str(model_file)+" ({}, {}, {}) - train, test score: \t {:.5f} - {:.5f} \t in {:.2f} s\n".format(hidden_layers , activation , alpha , train_score , test_score , train_time)


def load_preprocessed_data(wavelet = None):
    """ Load data that has already been processed into training data,
    testing data, training classifications, and testing
    classifications.

    filenames is a dictionary with keys Train.dat, Test.dat,
    Train.cls, and Test.cls
    """
    name_convert = {"Train.dat":"train_data", "Test.dat":"test_data",
                    "Train.cls":"train_classification",
                    "Test.cls":"test_classification"}
    data = {}
    for filename_raw, key_name in name_convert.items():
        if wavelet is not None:
            file_parts = filename_raw.split('.')
            filename = file_parts[0] + "." + wavelet + "." + file_parts[1]
        else:
            filename = filename_raw
        row_no = 0
        num_lines = sum(1 for line in open(filename))
        print("Loading data from " + filename + ".")
        with open(filename, 'r') as cur_file:
            cur_file_data = []
            csv_reader = csv.reader(cur_file, delimiter=",")
            for row in csv_reader:
                row_no += 1
                if row_no % 500 == 0:
                    progress_bar(row_no, num_lines)
                if len(row) == 1:
                    cur_file_data.append(int(row[0]))
                else:
                    cur_file_data.append([float(value) for value in row])
            progress_bar(row_no, num_lines)
            data[key_name] = cur_file_data
            print()
    return data

def get_nn_parameters():
    """ For now a dummy function with hardcoded parameters. """
    nn_params={
        'hidden_layer_size':[(100),(100,16),(50,32),(50,32,4)],
        'activation':['relu','tanh'],
        'alpha':[0.00001,0.0001,0.01]
    }
    return nn_params

def progress_bar(cur, total, width=55, chars=('#','_')):
    """ Print a progress bar. """
    bar_width = width - 5
    ratio = cur / total
    percent = int(ratio * 100 + .5)
    if percent < 10:
        percent_width = 1
    elif percent < 100:
        percent_width = 2
    else:
        percent_width = 3
    blanks = 4 - percent_width
    num_filled = int(ratio * bar_width + .5)
    num_empty = bar_width - num_filled
    print('\r' + chars[0] * num_filled + chars[1] * num_empty + ' ' * blanks +\
          str(percent) + '%', end='')




if __name__ == "__main__":
    print()
    nn_parameters = get_nn_parameters()
    process(nn_parameters)
