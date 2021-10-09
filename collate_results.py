""" Program to check the model description files built through multiple
directories and collate those results.
"""

import csv
import pickle
from os import walk
from os.path import join

def get_model_descs():
    """ Build a list of model_description files """
    model_descs = {}
    for (dirpath, dirnames, filenames) in walk('.'):
        for filename in filenames:
            if 'model_desc' in filename:
                numwavelets = None
                wavelet_name = filename.split('.')[1]
                if '-' in wavelet_name:
                    wavelet_name = wavelet_name.split('-')[0]
                dirs = str(dirpath).split('/')
                for direc in dirs:
                    if 'w' in direc:
                        numwavelets = int(direc.split('w')[1].split('.')[0])
                        break
                if numwavelets is None:
                    print(join(dirpath, filename) + " is skipped")
                else:
                    if numwavelets not in model_descs.keys():
                        model_descs[numwavelets] = {}
                    if wavelet_name not in model_descs[numwavelets].keys():
                        model_descs[numwavelets][wavelet_name] = {}
                    model_descs[numwavelets][wavelet_name][join(dirpath,\
                                                filename)] = str(dirpath) + "/"
    return model_descs

def extract_desc(desc_line, desc_filepath, numwavelets, include_mismatch=True, wavelet=None):
    """ Read model description file to get info. Returns a tab separated
    line with
    hidden_layers<TAB>act func<TAB>alpha<TAB>train score<TAB>test
    score<TAB>time to train<TAB>model_filename
    """
    line = desc_line
    model_filename = desc_filepath + line.split(" ")[0]
    params = line[line.find("(")+1:line.find('train')-4]
    if ")" in params:
        hidden_layers, remaining = params.split("), ")
        hidden_layers = hidden_layers + ")"
        activation, alpha = remaining.split(", ")
    else:
        hidden_layers, activation, alpha = params.split(", ")
        hidden_layers = "(" + hidden_layers + ")"
    scores, time = line.split("\t")[1:]
    train, test = scores[1:-1].split(" - ")
    time = time.split(" ")[2]
    outline = hidden_layers + "\t" + activation + "\t" + alpha + "\t" +\
            train + "\t" + test + "\t" + time + "\t" + model_filename
    if include_mismatch:
        #datafile = model_filename.split("NN")[0] + "Test." + wavelet + ".dat"
        datafile = "../wavelets-" + str(numwavelets) + "/Test." + wavelet + ".dat"
        #classfile = model_filename.split("NN")[0] + "Test." + wavelet + ".cls"
        classfile = "../wavelets-" + str(numwavelets) + "/Test." + wavelet + ".cls"
        outline += "\t" + str(run_model(model_filename, datafile, classfile))
    return outline

def run_model(model_file, data_file, class_file):
    """ Run the model and return a dictionary of actual:predicted. """
    output_counts={  # Actual:predicted
        0:{0:0,1:0,2:0,3:0,4:0},
        1:{0:0,1:0,2:0,3:0,4:0},
        2:{0:0,1:0,2:0,3:0,4:0},
        3:{0:0,1:0,2:0,3:0,4:0},
        4:{0:0,1:0,2:0,3:0,4:0}
    }
    classifier = pickle.load(open(model_file, 'rb'))
    input_data = []
    class_data = []
    with open(data_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            cur_data = [float(item) for item in row]
            input_data.append(cur_data)
    with open(class_file, 'r') as classes:
        for line in classes:
            class_data.append(int(float(line)))
    for index in range(len(class_data)):
        prediction = classifier.predict([input_data[index]])[0]
        actual = class_data[index]
        output_counts[actual][prediction] += 1
    return output_counts

def write_outfile(model_descs, outfilename):
    """ Write data to outfilename. Each line is as follows:
    # of wavelets<TAB>type of wavelet<TAB>hidden layers<TAB>activation<TAB>
    alpha<TAB>train score<TAB>test score<TAB>time to train<TAB>model
    filename<NEWL>
    """
    num_done = 0
    out_lines = []
    for num_wavelet, wavelet_types in model_descs.items():
        for wavelet_type, filenames in wavelet_types.items():
            for filename, filepath in filenames.items():
                with open(filename, 'r') as desc:
                    for line in desc:
                        out_line = str(num_wavelet) + "\t" + wavelet_type + "\t"
                        out_line = out_line + extract_desc(line, filepath,\
                                                           num_wavelet,\
                                                          True, wavelet_type)
                        out_lines.append(out_line)
                        num_done += 1
                        print("\rDone: " + str(num_done) + "   ", end='')
    with open(outfilename, 'w') as outfile:
        for line in out_lines:
            outfile.write(line + "\n")

if __name__ == "__main__":
    write_outfile(get_model_descs(), "data_summary.tsv")
