import random
import numpy as np
import util
from util import COL_LEN

TEST_DATA = "test.txt"
TRAINING_DATA = "classification_dataset_training.csv"
'''
The first column is reserved for ID and last column is the classification or output
'''

def seperate_train_validate_data(iteration, data_in, data_out ):
	trainData_in = []
	validationData_in = []
	for i in range(50):
		new_col = []
		trainData_in.append(new_col)
		validate_col = []
		validationData_in.append(validate_col)

	trainData_out = []
	validationData_out = []
	for j in range(50):
		validationData_in[j] = data_in[j][(500*iteration):(500*iteration+500)]
	validationData_out = data_out[(500*iteration):(500*iteration+500)]
	for j in range(50):
		trainData_in[j] = data_in[j][:(500*iteration)] + data_in[j][(500*iteration+500):]
	trainData_out = data_out[:(500*iteration)] + data_out[(500*iteration+500):]
	for i in range(len(trainData_out)):
		if trainData_out[i] == 0:
			trainData_out[i] = -1
	for i in range(len(trainData_in)):
		if validationData_out[i] == 0:
			validationData_out[i] = -1


	return trainData_in, trainData_out, validationData_in, validationData_out


def create_input_matrix(dataList):
	x = np.array([])
	x.resize(len(dataList[0]), COL_LEN-1)
	for i in range(len(dataList[0])):
		x.itemset((i, 0), 1)
		val = 1.0
		for j in range(COL_LEN-2):
			x.itemset((i, j+1), dataList[j][i] )
	return x

def find_single_output(w, iteration, validationData):
	val = w[0][0]
	for j in range(len(w)-1):
		val = val + w[j+1][0] * validationData[j][iteration]

	return val



def find_predicted_output(w, validationData):
	predicted_output = []
	for i in range(len(validationData[0])):
		val = find_single_output(w, i, validationData)
		if val < 0:
			predicted_output.append(-1)
		else:
			predicted_output.append(1)


	return predicted_output

def change_w(w, x, data_in, iteration):
	w[0][0] = w[0][0] + x
	for i in range(len(w)-1):
		w[i][0] = w[i][0] + x * data_in[i][iteration]

	return w


def run_single_instance_of_perceptron(w, data_in, data_out):

	for i in range(1):

		iteration_count = 0
		while(True):
			for i in range(len(data_in[0])):
				predicted = find_single_output(w, i, data_in)
				if ((predicted<0) and (data_out[i] > 0)):
					w = change_w(w, 0.5, data_in, i)
				else:
					if ( (predicted>=0) and (data_out[i] < 0)):
						w = change_w(w, -0.5, data_in, i)

			pred_input = find_predicted_output(w, data_in)
			count_val = util.count_mismatch(pred_input, data_out)
			print(count_val)
			iteration_count += 1
			if iteration_count >= 10:
				break

	return w

def run_linear_regression():
	input_col, output_col = util.read_file_data(TRAINING_DATA)
	min_error = 10000.0
	for i in range(1):
		trainData_in, trainData_out, validationData_in, validationData_out = seperate_train_validate_data(i, input_col, output_col)
		input_array = create_input_matrix(trainData_in)
		output_array = util.create_output_matrix(trainData_out)
		w = util.find_w(input_array, output_array)
		#pred_input = find_predicted_output(w.tolist(), trainData_in)
		w = run_single_instance_of_perceptron(w, trainData_in, trainData_out)
		print(w)
		pred_input = find_predicted_output(w, trainData_in)
		count_val = util.count_mismatch(pred_input, trainData_out)
		print("INITIAL COUNTER: " + str(count_val))
	print("")
	print("")


def unit_test():
	run_linear_regression()


if __name__ == "__main__":
	unit_test()
	#test_square_error()
	#run_algorithms()
