import scipy as sp
import numpy as np

COL_LEN = 52
ENABLE_LOG = False
TRAINING_DATA = "classification_dataset_training.csv"
TESTING_DATA = "classification_dataset_testing.csv"
TESTING_DATA_SOLUTION = "classification_dataset_testing_solution.csv"
VALIDATION_DATA_PERCENTAGE = 0.15
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


def find_w(x_array, y_array):
	x_transpose_array = x_array.transpose()
	x_xt_mult_array = np.dot(x_transpose_array, x_array)
	x_xt_mult_matrix = np.matrix(x_xt_mult_array)
	try:
		x_xt_mult_matrix_inv = x_xt_mult_matrix.getI()
	except np.linalg.linalg.LinAlgError:
		print("Singular Matrix Error detected")
		print(x_array)
		print(x_xt_mult_matrix)
	w_without_y = np.dot( x_xt_mult_matrix_inv, x_transpose_array)
	w = np.dot(w_without_y, y_array)
	return w


def create_output_matrix(dataList):
	y = np.array(([]))
	y.resize(1, len(dataList))
	for i in range(len(dataList)):
		y[0][i] = dataList[i]

	return y.transpose()


def read_file_data(file_name):
	file = open(file_name, "r")
	data_col = []
	output_col = []
	file.readline() #We don't need the first one
	first_time = True
	for line in file:
		strData = line.split(",");
		for i in range(COL_LEN-2):
			if first_time:
				new_col = []
				data_col.append(new_col)
			data_col[i].append(int(strData[i+1]))
		output_col.append(int(strData[COL_LEN-1]))
		first_time = False

	file.close()
	return data_col, output_col



def read_file_header(file_name):
	file = open(file_name, "r")
	data_col = []
	output_col = []
	line = file.readline() #We don't need the first one
	words = line.split(",")

	return words[1:51]

def read_file_data_rowmajor(file_name):
	file = open(file_name, "r")
	data_col = []
	output_col = []
	file.readline() #We don't need the first one
	first_time = True
	for line in file:
		strData = line.split(",");
		data_tmp = []
		for i in range(COL_LEN-2):
			data_tmp.append(int(strData[i+1]))
		data_col.append(data_tmp)
		output_col.append(int(strData[COL_LEN-1]))
		first_time = False

	file.close()
	return data_col, output_col

def read_file_data_rowmajor_test(input, output):
	file = open(input, "r")
	data_col = []
	output_col = []
	file.readline() #We don't need the first one
	for line in file:
		strData = line.split(",");
		data_tmp = []
		for i in range(len(strData) - 1):
			data_tmp.append(int(strData[i+1]))
		data_col.append(data_tmp)

	file.close()
	file = open(output, "r")
	file.readline()
	for line in file:
		strData = line.split(",")
		output_col.append(int(strData[1]))
	file.close()
	return data_col, output_col


def read_file_data_rowmajor_(i1, i2, file_name):
	file = open(file_name, "r")
	data_col = []
	output_col = []
	file.readline() #We don't need the first one
	first_time = True
	for line in file:
		strData = line.split(",");
		data_tmp = []
		for i in range(COL_LEN-2):
			if i != i1 and i!= i2:
				data_tmp.append(int(strData[i+1]))
		data_col.append(data_tmp)
		output_col.append(int(strData[COL_LEN-1]))
		first_time = False

	file.close()
	return data_col, output_col

def count_mismatch(pred, validationData_out):
	match = 0
	for i in range(len(pred)):
		if( (pred[i] < 0.5 and validationData_out[i] < 0.5)or(pred[i] >= 0.5 and validationData_out[i] >= 0.5) ):
			match += 1
	return match

def get_output_data( data_out ):
	data_len = len(data_out)
	train_data_len = int(data_len - VALIDATION_DATA_PERCENTAGE * data_len)
	trainData_out = []
	validationData_out = []

	##First 4000 data are taken for feature selection and next 500 data for validation 
	validationData_out = data_out[train_data_len:data_len]
	trainData_out = data_out[:train_data_len]

	return trainData_out, validationData_out