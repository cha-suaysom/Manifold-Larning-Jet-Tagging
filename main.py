from data_loader_shuffler_training import *
from data_loader_shuffler_testing import *
from manifold_learning_training import *
from manifold_learning_testing import *
import logging
import sys
import argparse

if __name__ == '__main__':
	# Enable logging
	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

	parser = argparse.ArgumentParser(description = "Manifold learning on jet tagging")
	parser.add_argument('--steps', metavar = "steps", type = str, default = "1,2,3,4,5", nargs = '?', help = 'Steps to do in the code')

	args = parser.parse_args()
	steps_list = args.steps.split(',')
	print(steps_list)

	# Load data with the following file names
	if ('1' in steps_list):
		training_filename = "data/train.h5"
		testing_filename = "data/test.h5"
		num_sample = 2
		num_data = 5000
		if (load_shuffle_training_data(training_filename, num_sample,num_data,test = False) and
			load_shuffle_testing_data(testing_filename, num_sample,num_data,test = False)):
			logging.info("Load data successfully!")
		else:
			logging.error("Cannot load data successfully")

	if ('2' in steps_list):
		step2()
	if ('3' in steps_list):
		step3()