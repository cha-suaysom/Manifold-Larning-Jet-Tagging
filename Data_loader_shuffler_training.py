import numpy as np

import pandas

#num_test = 3000

input_filename = "train.h5"
training_set = pandas.HDFStore(input_filename)
#training_set = training_set.select("table",stop = 3*num_test)
training_set = training_set.select("table")

# top_training_set = training_set[training_set["is_signal_new"] == 1][:num_test]
# qcd_training_set = training_set[training_set["is_signal_new"] == 0][:num_test]

# top_training_set = np.array(top_training_set.iloc[:, :800])
# qcd_training_set = np.array(qcd_training_set.iloc[:,:800])

import numpy as np

# num_files = num_test
# A_all = np.zeros((num_files,800))
# B_all = np.zeros((num_files,800))

# for j in range(num_files):
#     #Reading the 100 files   
#     A_all[j] = top_training_set[j]
#     B_all[j] = qcd_training_set[j]

len(training_set)

top_training_set = training_set[training_set["is_signal_new"] == 1]
qcd_training_set = training_set[training_set["is_signal_new"] == 0]

top1=top_training_set.sample(n=5000)

top2=top_training_set.sample(n=5000)

top1

top2

np.sum(top1["PX_0"])

np.sum(top2["PX_0"])

import pickle
num_sample = 10
num_data = 50000
for i in range(num_sample):
    sample_top_training = top_training_set.sample(n=num_data)
    file = open('sample_data/sample_data_train_top'+str(i), 'wb')
    # dump information to that file
    pickle.dump(sample_top_training, file)
    # close the file
    file.close()

import pickle
num_sample = 10
num_data = 50000
for i in range(num_sample):
    sample_qcd_training = qcd_training_set.sample(n=num_data)
    file = open('sample_data/sample_qcd_train_top'+str(i), 'wb')
    # dump information to that file
    pickle.dump(sample_qcd_training, file)
    # close the file
    file.close()