import random
import numpy as np
import util
from util import COL_LEN, ENABLE_LOG, TRAINING_DATA

TEST_DATA = "test.txt"
list_of_features = []
'''
The first column is reserved for ID and last column is the classification or output
'''

def append_new_column(iteration, train_data_in, validation_data_in, data_in ):
	data_len = len(data_in[0])
	train_data_len = int(data_len - util.VALIDATION_DATA_PERCENTAGE * data_len)
	return_train_data_in = []
	return_validate_data_in = []
	for data in train_data_in:
		return_train_data_in.append(data)
	return_train_data_in.append(data_in[iteration][:train_data_len])
	for data in validation_data_in:
		return_validate_data_in.append(data)
	return_validate_data_in.append(data_in[iteration][train_data_len:data_len])

	return return_train_data_in, return_validate_data_in



def create_selected_column_matrix( data_in, selected_column ):
	selected_columns = []
	print(len(data_in), len(data_in))
	for row in selected_column:
		selected_columns.append(data_in[row][:len(data_in[0])])
	return selected_columns

def create_input_matrix(dataList):
	x = np.array([])
	x.resize(len(dataList[0]), len(dataList) + 1)
	for i in range(len(dataList[0])):
		x.itemset((i, 0), 1)
		for j in range(len(dataList)):
			x.itemset((i, j+1), dataList[j][i])

	return x

def find_predicted_output(w, validationData):
	predicted_output = []
	for i in range(len(validationData[0])):
		val = w[0][0]
		for j in range(len(w) - 1):
			val = val + validationData[j][i] * w[j+1][0]
		predicted_output.append(val)

	return predicted_output

def convert_to_binary(output_data):
	binary_output_data = []
	for data in output_data:
		if data >= 0.5 :
			binary_output_data.append(1)
		else:
			binary_output_data.append(0)

	return binary_output_data

def get_w(input_data, output_data):
	input_array = create_input_matrix(input_data)
	output_array = util.create_output_matrix(output_data)
	w = util.find_w(input_array, output_array)

	return w


def run_linear_regression(input_col, output_col):
	list_pos = []
	list_w = []
	trainData_out, validationData_out = util.get_output_data(output_col)
	trainData_in = []
	validationData_in = []
	min_error_list = []
	max_iter_count = 0.0
	prev_iter_count = 0.0
	mininum_reached = False
	#CHANGE IT BACK TO 50
	for i in range(50):
		min_error = 10000.0
		best_pos = -1
		best_w = 0
		for j in range(50):
			found = False
			for x in list_pos:
				if x == j:
					found = True
			if found:
				continue
			temp_train_in, temp_validation_in = append_new_column( j, trainData_in, validationData_in, input_col)

			w = get_w(temp_train_in, trainData_out)
			pred_intim = find_predicted_output(w.tolist(), temp_validation_in)
			pred = convert_to_binary(pred_intim)
			count_ = util.count_mismatch( pred, validationData_out)

			val = util.logloss(pred, validationData_out)
			if val < min_error:
				max_iter_count = count_
				min_error = val
				best_pos = j
		
		if prev_iter_count > max_iter_count:
			break
		prev_iter_count = max_iter_count
		list_pos.append(best_pos)
		min_error_list.append(min_error)
		trainData_in, validationData_in = append_new_column( best_pos, trainData_in, validationData_in, input_col)
		if ENABLE_LOG:
			print(max_iter_count, len(trainData_in))


	if ENABLE_LOG:
		print(list_pos)
		print(min_error_list)
		print("")
		print("")
		print("")

	print(len(input_col), len(input_col[0]))
	w = get_w(trainData_in, trainData_out)
	#input_col[:][len(input_col) - int(len(input_col)*util.VALIDATION_DATA_PERCENTAGE)]
	train_len = len(input_col[0]) - int(len(input_col[0])*util.VALIDATION_DATA_PERCENTAGE)
	'''
	test_list = []
	for i in range(len(input_col)):
		test_list.append(input_col[i][:train_len])
	'''
	pred = test(w, list_pos, input_col)
	if ENABLE_LOG:
		count_ = util.count_mismatch( pred, output_col)
		print(count_)

	return pred, w, list_pos

#Data should not be in Row major format
def test(w, selected_pos, data_in):
	selected_matrix = create_selected_column_matrix( data_in, selected_pos)
	pred_intim = find_predicted_output(w.tolist(), selected_matrix)
	pred = convert_to_binary(pred_intim)

	return pred


def unit_test():
	input_col, output_col = util.read_file_data(TRAINING_DATA)
	_, _, _ = run_linear_regression(input_col, output_col)


if __name__ == "__main__":
	unit_test()