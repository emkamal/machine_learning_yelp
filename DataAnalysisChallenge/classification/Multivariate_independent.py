import numpy as np
import util
import math


TRAINING_DATA = "classification_dataset_training.csv"
'''
1. Find the mean using the libraray
2. Implement a method for finding the variance using the library
'''

'''
Try to use regularizer here to make it better. Next, chpater in cross validation from the Caltech may help
with it.
'''
def mean(input_arr, axis):
	return np.mean(input_arr, axis=axis)

def variance(input_arr, axis):
	return np.var(input_arr, axis=axis)

def calculate_prob(output_arr, val ):
	count = 0
	if len(output_arr) < 0:
		return;
	for i in output_arr:
		if i == val:
			count += 1
	return count/len(output_arr)

def classify_data_to_classes(input_arr, output_arr):
	class_data_0 = []
	class_data_1 = []
	for i in range(len(output_arr)):
		if(output_arr[i] == 0 ):
			class_data_0.append(input_arr[i])
		else:
			class_data_1.append(input_arr[i])
	return class_data_0, class_data_1

def separate_train_validate(input_col, output_col):
	train_data_in = input_col[:4000][:]
	train_data_out = output_col[:4000]
	valid_data_in = input_col[4000:4500][:]
	valid_data_out = output_col[4000:4500]

	return train_data_in, valid_data_in, train_data_out, valid_data_out

def find_distribution_strength(data, mean, var):
	sum = 0.0
	for j in range(len(mean)):
		sum = sum + (data[j] - mean[j])*(data[j] - mean[j])/var[j]
	return (-sum/2)
		
def classify_validation_data(valid_data_in, p0, p1, mean_0, mean_1, var_0, var_1):
	predicted_output = []
	cnst_0 = np.log(p0)
	cnst_1 = np.log(p1)
	for i in range(len(valid_data_in)):
		sum_0 = find_distribution_strength(valid_data_in[i], mean_0, var_0)
		sum_1 = find_distribution_strength(valid_data_in[i], mean_1, var_1)
		if( (cnst_0 + sum_0) > (cnst_1 + sum_1) ):
			predicted_output.append(0)
		else:
			predicted_output.append(1)

	return predicted_output


def main_indpendent_var():

	input_col, output_col = util.read_file_data_rowmajor(TRAINING_DATA)
	train_data_in, valid_data_in, train_data_out, valid_data_out = separate_train_validate( input_col, output_col)

	class_data_0, class_data_1 = classify_data_to_classes(train_data_in, train_data_out)
	mean_0 = mean(class_data_0, axis=(0))
	var_0 = variance(class_data_0, axis=(0))

	mean_1 = mean(class_data_1, axis=(0))
	var_1 = variance(class_data_1, axis=(0))

	p0 = calculate_prob(train_data_out, 0)
	p1 = calculate_prob(train_data_out, 1)
	predicted_output = classify_validation_data(valid_data_in, p0, p1, mean_0, mean_1, var_0, var_1)

	count_val = util.count_mismatch(predicted_output, train_data_out)
	val = util.logloss(predicted_output, valid_data_out)
	print(val)
	print(count_val, len(predicted_output))

def get_covar_matrix(data):
	data_array = np.transpose(np.array(data))
	covar_matrix = np.cov(data_array)

	return covar_matrix

def calculate_value(input_row, mean, covar):
	input_array = np.reshape(np.array(input_row), (len(input_row), 1))
	mean_array = np.reshape(np.array(mean), (len(mean), 1))
	inv_covar = []
	try:
		inv_covar = np.linalg.inv(covar)
	except np.linalg.LinAlgError:
		print("SOMETHING HORRIBLE WENT WRONG WHEN TRYING TO INVERT THE MATRIX")
		pass
	else:
		diff_x_mu = np.subtract(input_array, mean_array)
		mult_temp = np.dot(np.transpose(diff_x_mu), inv_covar)
		final_mult = np.dot(mult_temp, diff_x_mu)
		final_mult = final_mult/2.0
		exp_val = np.exp(-final_mult)
		deter = np.linalg.det(covar)
		if( deter < 0):
			deter = -deter
		deter_value = math.sqrt(deter)
		pi_factor = math.pow(2*math.pi, len(input_row)/2)
		divider = deter_value * pi_factor
		strength = exp_val/divider

	return strength

def classify_validate_data(valid_data_in, p0, p1, mean_0, mean_1, covar_0, covar_1):
	predicted_output = []
	for data in valid_data_in:
		val_0 = calculate_value(data, mean_0, covar_0)
		val_1 = calculate_value(data, mean_1, covar_1)
		if( (val_0 + p0) > (val_1 + p1)):
			predicted_output.append(0)
		else:
			predicted_output.append(1)

	return predicted_output

def main_dependent_var():
	for i in range(50):
		input_col, output_col = util.read_file_data_rowmajor_(i, i+1, TRAINING_DATA)
		train_data_in, valid_data_in, train_data_out, valid_data_out = separate_train_validate( input_col, output_col)

		class_data_0, class_data_1 = classify_data_to_classes(train_data_in, train_data_out)
		mean_0 = mean(class_data_0, axis=(0))
		var_0 = variance(class_data_0, axis=(0))

		mean_1 = mean(class_data_1, axis=(0))
		var_1 = variance(class_data_1, axis=(0))

		p0 = calculate_prob(train_data_out, 0)
		p1 = calculate_prob(train_data_out, 1)

		covar_0 = get_covar_matrix(class_data_0)
		#print(len(class_data_0), len(class_data_0[0]))
		covar_1 = get_covar_matrix(class_data_1)

		predicted_output = classify_validate_data(valid_data_in, p0, p1, mean_0, mean_1, covar_0, covar_1)
		print(len(predicted_output))

		count_val = util.count_mismatch(predicted_output, train_data_out)
		val = util.logloss(predicted_output, valid_data_out)
		print(count_val, val)


if __name__ == '__main__':
	main_dependent_var()