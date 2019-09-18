import numpy as np
import pickle
import pandas
import logging
import os.path
import pathlib

def load_shuffle_training_data(input_filename, num_sample,num_data, test = True):
    if (os.path.exists(input_filename)):
        training_set = pandas.HDFStore(input_filename)
    #logging.info("Load the data succesfully")
        logging.info("Load training data successfully")
    else:
        logging.error("The training file " + input_filename + " does not exist. Please download them and put them in the correct directory per the directions in the README")
        #return False
        return False

    try:
        training_set = training_set.select("table")
        top_training_set = training_set[training_set["is_signal_new"] == 1]
        qcd_training_set = training_set[training_set["is_signal_new"] == 0]
        logging.info("Successfully Obtain the relevant top and qcd data")
    except:
        logging.error("Please check the validity of the dataset. Does " + input_filename + " contain corrects .h5 columns and is obtained correctly?")
        return False

    if (test):
        num_data/= 100 # For testing, only use 1 percent of the data

    for i in range(num_sample):
        sample_top_training = top_training_set.sample(n=num_data)
        abspath = pathlib.Path('sample_data/sample_data_training_top'+str(i)).absolute()
        file = open(str(abspath), 'wb')
        pickle.dump(sample_top_training, file)
        file.close()

    for i in range(num_sample):
        sample_qcd_training = top_training_set.sample(n=num_data)
        abspath = pathlib.Path('sample_data/sample_data_training_qcd'+str(i)).absolute()
        file = open(str(abspath), 'wb')
        pickle.dump(sample_qcd_training, file)
        file.close()
    return True