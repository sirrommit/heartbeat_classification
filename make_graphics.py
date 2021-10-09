""" Program to produce graphics for a write-up of the ECG cardiac
classification program.

Copyright (c) 2021 Timothy Morris Phd
"""

import csv
import operator
import numpy as np
import matplotlib.pyplot as plt

def read_data_file(filename):
    """ Read filename and produce a list of data. """
    file_order=['num_wavelets', 'wavelet_type', 'hidden_layers',
                'activation_func', 'alpha', 'train_score', 'test_score',
                'train_time', 'model_file', 'mismatch']
    conversion_func = {'num_wavelets':int, 'wavelet_type':str,
                      'hidden_layers':read_tuple, 'activation_func':str,
                      'alpha':float, 'train_score':float, 'test_score':float,
                      'train_time':float, 'model_file':str,
                      'mismatch':read_mismatch}

    out_list = []

    with open(filename, 'r') as data_file:
        csv_reader = csv.reader(data_file, delimiter='\t')
        for row in csv_reader:
            row_dict = {}
            for index in range(len(file_order)):
                key_name = file_order[index]
                row_dict[key_name] = conversion_func[key_name](row[index])
            out_list.append(row_dict)
    return out_list

def read_tuple(tuple_string):
    """ Take a string and convert it to a tuple of integers. """
    tuple_string = tuple_string[1:-1]  # Remove ( and ) from ends
    if tuple_string[-1] == ',':
        tuple_string = tuple_string[:-1]
    return tuple([int(item) for item in tuple_string.split(', ')])

def read_mismatch(mis_string, convert=True):
    """ Take the mismatch string from the data file and turn it into a
    dictionary.
    """
    if convert:
        conversion = {0:'N', 1:'S', 2:'V', 3:'F', 4:'Q'}
    else:
        conversion = {0:0, 1:1, 2:2, 3:3, 4:4}
    mis_string = mis_string[1:-3]  # Remove { and }}\n from ends
    out_dict = {conversion[index]:{conversion[inner]:0 for inner in
                                   conversion.keys()} for index in
                conversion.keys()}  # Set up out_dictionary with zeroes
    key_items = mis_string.split('}, ')
    for key_item in key_items:
        outer_key = int(key_item.split(": {")[0])
        inner_string = key_item.split(": {")[1]
        for inner_item in inner_string.split(", "):
            inner_key, number = inner_item.split(": ")
            inner_key = int(inner_key)
            number = int(number)
            out_dict[conversion[outer_key]][conversion[inner_key]] = number
    return out_dict

def filter_data(raw_data, filter_list, order_by=None, descending=False):
    """ Return sublist of raw_data that only includes rows that meet all of
    the filters in the filter_list.
    Each filter in filter_list is a list [key, operator, value]
    """
    out_values = []
    for row in raw_data:
        include_row = True
        for filt in filter_list:
            if not filt[1](row[filt[0]], filt[2]):
                include_row = False
        if include_row:
            out_values.append(row)
    if order_by is not None:
        ordered = []
        for row in out_values:
            index = 0
            while index < len(ordered) and row[order_by] >\
                  ordered[index][order_by]:
                index += 1
            ordered.insert(index, row)
        if descending:
            return ordered[::-1]
    return out_values

def line_plots(x_vals, x_label, y_vals, y_label, title, legend=None , save=None,
                 show=True):
    """ Make a connected scatter plot with 1 or more y-series on a single x axis """
    fig, ax = plt.subplots()
    for index, y_s in enumerate(y_vals):
        if legend is not None:
            ax.plot(x_vals, y_s, label=legend[index])
        else:
            ax.plot(x_vals, y_s)
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.grid()
    if legend is not None:
        ax.legend(loc="lower right")
    if save is not None:
        fig.savefig(save)
    if show:
        plt.show()

def scatter_plots(x_vals, x_label, y_vals, y_label, title, legend=None, save=None,
                 show=True):
    """ Make a connected scatter plot with 1 or more y-series on a single x axis """
    fig, ax = plt.subplots()
    for index, y_s in enumerate(y_vals):
        if legend is not None:
            ax.scatter(x_vals[index], y_s, label=legend[index])
        else:
            ax.scatter(x_vals[index], y_s)
    if legend is not None:
        ax.legend(loc="lower right")
    ax.plot([.996,.9995],[.996,.9995])
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    if save is not None:
        fig.savefig(save)
    if show:
        plt.show()

def num_layers(layer_desc, num):
    """ Return true is layer_desc has num elements in the tuple. """
    if len(layer_desc) == num:
        return True
    return False

if __name__ == "__main__":
    data_list = read_data_file('data_summary.tsv')
    filtered = filter_data(data_list, [['test_score', operator.gt, .99936]],
                          order_by='test_score', descending=True)
    for row in filtered:
        print(row['test_score'], row['num_wavelets'], row['wavelet_type'],
              row['hidden_layers'], row['activation_func'], row['alpha'])
        mismatch = row['mismatch']
        print("\t",end='')
        for pos in mismatch.keys():
            print(pos,end='\t')
        print()
        for actual, predicted in mismatch.items():
            print(actual,end='\t')
            for pred, num in predicted.items():
                print(num,end='\t')
            print()
        print()




#    x_vals = np.arange(1, 101, 1)
#    y_vals = []
#    runs = []
#    runs.append(filter_data(data_list, [['wavelet_type', operator.eq, 'morl'],
#                                   ['num_wavelets', operator.eq, 10]]))
#    runs.append(filter_data(data_list, [['wavelet_type', operator.eq, 'gaus5'],
#                                   ['num_wavelets', operator.eq, 25]]))
#    runs.append(filter_data(data_list, [['wavelet_type', operator.eq, 'morl'],
#                                   ['num_wavelets', operator.eq, 25],
#                                   ['hidden_layers', num_layers, 1]]))
#    runs.append(filter_data(data_list, [['wavelet_type', operator.eq, 'morl'],
#                                   ['num_wavelets', operator.eq, 50],
#                                   ['activation_func', operator.eq, 'tanh']]))
#    runs.append(filter_data(data_list, [['wavelet_type', operator.eq, 'morl'],
#                                   ['num_wavelets', operator.eq, 50],
#                                   ['activation_func', operator.eq, 'relu']]))
#    runs.append(filter_data(data_list, [['wavelet_type', operator.eq, 'morl'],
#                                   ['num_wavelets', operator.eq, 25],
#                                   ['hidden_layers', num_layers, 2]]))
#    legend = ["10 morl with X hidden",\
#              "25 gaus5 with X hidden",\
#              "25 morl with X hidden",\
#              "50 morl with X-16 hidden (tanh)",\
#              "50 morl with X-16 hidden (relu)",\
#              "25 morl with X-32 hidden"]
#    for run in runs:
#        run_y = []
#        for row in run:
#            run_y.append(row['test_score'])
#        y_vals.append(np.array(run_y))
# 
#    line_plots(x_vals, "hidden layer size", y_vals, "test data accuracy",
#                  "Affect of layer size on Classification Success", legend,
#                  save="incremental_convergence.png", show=False)


#    x_vals = []
#    y_vals = []
# 
#    runs = []
#    runs.append(filter_data(data_list, [['wavelet_type', operator.eq, 'morl'],
#                                   ['num_wavelets', operator.eq, 10]]))
#    runs.append(filter_data(data_list, [['wavelet_type', operator.eq, 'gaus5'],
#                                   ['num_wavelets', operator.eq, 25]]))
#    runs.append(filter_data(data_list, [['wavelet_type', operator.eq, 'morl'],
#                                   ['num_wavelets', operator.eq, 25],
#                                   ['hidden_layers', num_layers, 1]]))
#    runs.append(filter_data(data_list, [['wavelet_type', operator.eq, 'morl'],
#                                   ['num_wavelets', operator.eq, 25],
#                                   ['hidden_layers', num_layers, 2]]))
#    legend = ["10 morl with 100-16 hidden",\
#              "25 gaus5 with 100 hidden",\
#              "25 morl with 100 hidden",\
#              "25 morl with 50-32 hidden"]
#    for run in runs:
#        run_x = []
#        run_y = []
#        for row in run:
#            run_x.append(row['train_score'])
#            run_y.append(row['test_score'])
#        x_vals.append(np.array(run_x))
#        y_vals.append(np.array(run_y))
#    scatter_plots(x_vals, "Accuracy on Training Data", y_vals, "Accuracy on Test Data",
#              "Comparing Training and Testing Accuracy", legend, save="reproducible.png")

#    ax = [None for x in range(4)]
#    fig, ((ax[0], ax[1]), (ax[2], ax[3])) = plt.subplots(2, 2)
#    for index, numwave in enumerate([10, 25, 50, 75]):
#        y_vals = []
#        filtered = filter_data(data_list, [['num_wavelets', operator.eq, numwave]])
#        for row in filtered:
#            y_vals.append(row['test_score'])
#        num_bins = 10
#        n, bins, patches = ax[index].hist(y_vals, num_bins, density=False)
#        ax[index].set_xlabel("Prediction Rate")
#        ax[index].set_ylabel("Probability density")
#        ax[index].set_title(str(numwave) + " wavelets.")
#    fig.suptitle("Effect of Parameter Choice on Training by number of wavelets.")
#    fig.tight_layout()
#    fig.savefig("parameter_tuning.png")

