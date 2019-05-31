import numpy as np

import pandas

#num_test = 3000

input_filename = "test.h5"
testing_set = pandas.HDFStore(input_filename)
#testing_set = testing_set.select("table",stop = 3*num_test)
testing_set = testing_set.select("table")

# top_testing_set = testing_set[testing_set["is_signal_new"] == 1]
# qcd_testing_set = testing_set[testing_set["is_signal_new"] == 0]

# top_testing_set = np.array(top_testing_set.iloc[:, :800])
# qcd_testing_set = np.array(qcd_testing_set.iloc[:,:800])

top_testing_set = testing_set[testing_set["is_signal_new"] == 1]
qcd_testing_set = testing_set[testing_set["is_signal_new"] == 0]

len(top_testing_set)

len(qcd_testing_set)

import pickle
num_sample = 10
num_data = 50000
for i in range(num_sample):
    sample_top_testing = top_testing_set.sample(n=num_data)
    file = open('sample_data/sample_data_testing_top'+str(i), 'wb')
    # dump information to that file
    pickle.dump(sample_top_testing, file)
    # close the file
    file.close()

import pickle
num_sample = 10
num_data = 50000
for i in range(num_sample):
    sample_qcd_testing = qcd_testing_set.sample(n=num_data)
    file = open('sample_data/sample_data_testing_qcd'+str(i), 'wb')
    # dump information to that file
    pickle.dump(sample_qcd_testing, file)
    # close the file
    file.close()