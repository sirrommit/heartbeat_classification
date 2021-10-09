#!/usr/bin/env python3
""" Defines wavelet processing function and allows user exploration with
wavelets.
"""

import csv
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pywt
import pandas as pd


def remove_trailing_zeroes(data_line):
    """
    The data files add a bunch of trailing
    zeroes so that each row has the same number of columns. This is
    not helpful for what we are doing, so I am removing the trailing
    zeroes.
    """
    trailing = not bool(data_line[-1]) # Check if last element is 0
    while trailing:
        data_line = data_line[0:-1] # replace the list with the a list
                                    # containing the first n-1 elements
        if data_line[-1] != 0:
            trailing=False
    return data_line

def scale_one_dim(signal, out_size):
    """ Scale a quantized signal so that it has exactly out_size values
    in it. Use a line between successive points to estimate values
    between given data. This is designed to scale down a signal to less
    values than are given originally.
    """
    sig_dim = len(signal)
    if out_size == sig_dim:
        return signal
    delta_out = (sig_dim - 1) / (out_size - 1) # outsize-1 is number of gaps
                                               # sig_dim-1 excludes last
                                               # datapoint so that first
                                               # and last can be included
    out_signal = []
    for step in range(out_size - 1):
        x_val = step * delta_out
        x_1 = int(x_val)
        dist = x_val - x_1
        diff = signal[x_1+1]-signal[x_1]
        out_signal.append(signal[x_1]+diff*dist)
    out_signal.append(signal[-1])
    return out_signal

def wavelet_transform(time, signal, scales, waveletname, x_points=None):
    """ Transform data by performing a wavelet decomposition
    time is a list of time values
    signal is the signal to decompose
    scales is the collection of "wavelengths" to use
    waveletname is the type of wavelet
    x_points is the number of points to scale each time row by
    """
    dtime = time[1]-time[0] # define time differential
    [coefficients,frequencies] = pywt.cwt(signal, scales, waveletname, dtime)
    scaled_coefficients=[]
    if x_points is not None:
        for co_row in coefficients: # Scale each row individually
            scaled_coefficients.append(scale_one_dim(co_row, x_points))
        return scaled_coefficients, scale_one_dim(frequencies,x_points)
    return coefficients, frequencies

def linearize(wavelet_decomp):
    """ Take a two-dimentsional wavelet decomposition and place each value
    in a single list.
    """
    linearized=[]
    for co_row in wavelet_decomp:
        for col in co_row:
            linearized.append(col)
    return linearized

def plot_signal_wavelet(time, signal, scales, classification, ecg_id, \
                        waveletname='mexh', cmap="seismic", \
                        ylabel='Period', xlabel='Time', scale=None):
    """ Plots wavelet decomposition with signal below it """
    title = 'Wavelet('+waveletname+') Transform (Power Spectrum) of ECG(' +\
            str(ecg_id)+')'

    # wavelet

    coefficients, frequencies = wavelet_transform(time, signal, scales,
                                                  waveletname, x_points=scale)
    power = (abs(coefficients)) ** 2
    period = 1.0 / frequencies
    minpower=power[0][0]
    maxpower=power[0][0]
    for column in power:
        for pow_val in column:
            if pow_val < minpower:
                minpower = pow_val
            if pow_val > maxpower:
                maxpower = pow_val
    st_pow2 = 0
    while 2.0**st_pow2 > minpower:
        st_pow2 = st_pow2-1
    end_pow2 = 0
    while 2.0**end_pow2 < maxpower:
        end_pow2 = end_pow2 + 1
    levels = []
    if st_pow2 < -30:
        st_pow2 = -30
    if end_pow2>30:
        end_pow2=30
    for pow_val in range(st_pow2,end_pow2):
        levels.append(2.0**pow_val)
    contourlevels = np.log2(levels)

    fig = plt.figure(figsize=(15,10))
    grid_spec = gridspec.GridSpec(2,1,height_ratios=[3,1]) # Define height of two graphs
    ax1 = plt.subplot(grid_spec[0])
    image = ax1.contourf(time,np.log2(period), np.log2(power), contourlevels,
                     cmap=cmap)
    ax1.set_title(title, fontsize=20)
    ax1.set_ylabel(ylabel, fontsize=18)

    yticks = 2**np.arange(np.ceil(np.log2(period.min())),
                          np.ceil(np.log2(period.max())))
    ax1.set_yticks(np.log2(yticks))
    ax1.set_yticklabels(yticks)
    ax1.invert_yaxis()
    ylim = ax1.get_ylim()
    ax1.set_ylim(ylim[0],2)

    cbar_ax = fig.add_axes([0.925, 0.5,.02, .25])
    fig.colorbar(image, cax=cbar_ax, orientation='vertical')

    # signal

    ax2 = plt.subplot(grid_spec[1])
    ax2.plot(time,signal,label='signal')
    ax2.set_xlim([time[0],time[-1]])
    ax2.set_xlabel(xlabel, fontsize=18)
    ax2.set_ylabel('Signal Amplitude', fontsize=18)
    ax2.set_title('ECG (classification='+classification+')', fontsize=20)
    ax2.legend()
    plt.show()

def show_selected(data, dataclass, data_key, wavelet):
    """ function to call function that shows plots """
    df_data = pd.DataFrame(data[data_key])
    n_val = df_data.shape[0]
    time0 = 0
    dtime = 1
    time = np.arange(0,n_val)*dtime+time0
    signal = df_data.values.squeeze()

    scales = np.arange(1,64) #[1,2,...,63]
    try:
        plot_signal_wavelet(time,signal,scales,dataclass[data_key],data_key,wavelet,None)
    except:
        print("Error: check wavelet name")

def load_data(dataset, convert_class=True):
    """ Load data from file=dataset """
    data = []
    dataclass = []
    convert = {0:'N', 1:'S', 2:'V', 3:'F', 4:'Q'}
    # Count number of lines in data
    num_lines = sum(1 for line in open(dataset))

    # Read data into a list
    with open(dataset,mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count%2000 == 0:
                print("\rProcessing Data: "+str(line_count/num_lines*10000//10/10)+" %",end="")
            cur_data = [float(d_item) for d_item in row[:-1]]
            if convert_class:
                cur_class = convert[int(row[-1].split('.')[0])]
            else:
                cur_class = int(row[-1].split('.')[0])
            data.append(remove_trailing_zeroes(cur_data))
            dataclass.append(cur_class)
            line_count=line_count+1
        print()
    return data, dataclass

def interactive_visualization(dataset):
    """ Function to visualize interactively """
    wavelet='morl'
    data, dataclass = load_data(dataset)

    data_keys=list(range(len(data)))

    key_no = -1

    # Interact with the data.
    while True:
        select = input("0 - "+str(len(data_keys)-1)+", N("+str(key_no+1)+")=next, P("+str(key_no-1)+")=previous, R=random, Q=quit, W=Change Wavelet: ")
        if select in ['Q', 'q']:
            break
        if select in ['N', 'n']:
            key_no = key_no + 1
            if key_no < len(data_keys):
                show_selected(data, dataclass, key_no, wavelet)
            else:
                print("Already at last ECG")
        elif select in ["P", 'p']:
            key_no = key_no - 1
            if key_no >= 0:
                show_selected(data, dataclass, key_no, wavelet)
            else:
                print("Already at first ECG")
        elif select in ["R", 'r']:
            key_no = randint(0,len(data_keys)-1)
            show_selected(data, dataclass, key_no, wavelet)
        elif select in ["W", 'w']:
            print("Some possible wavelets: mexh,morl,cgau5,gaus5")
            wavelet = input("Wavelet name: ")
        else:
            try: # If the input is an integer
                selint=int(select)
                if selint >= 0 and selint < len(data_keys):
                    key_no = selint
                    show_selected(data, dataclass, key_no, wavelet)
            except: # If the input is not an integer ignore it
                pass

if __name__ == "__main__":
    # Change these to select your data source.
    DATASET = "mitbih_train.csv"
    #DATASET = "mitbih_test.csv"
    #DATASET = "ptbdb_abnormal.csv"
    #DATASET = "ptbdb_normal.csv"
    interactive_visualization(DATASET)
