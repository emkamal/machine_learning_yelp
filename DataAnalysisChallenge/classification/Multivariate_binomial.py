import numpy as np
import util
import math
from util import ENABLE_LOG, TRAINING_DATA

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

def calculate_prio(arr):
	count = 0
	for i in arr:
		if i == 1:
			count += 1

	return (len(arr)-count)/len(arr), count/len(arr)

#BAG OF WORD MODELS
def calculate_probability_for_each_dimension(arr, output): #500x50
	countij_0 = []
	countij_1 = []
	count_0 = 0
	count_1 = 0
	for i in range(50):
		countij_0.append(0)
		countij_1.append(1)
	for i in range(len(output)):
		if output[i] > 0:
			count_1 += 1
			for j in range(len(arr[0])):
				countij_1[j] = countij_1[j] + arr[i][j]
		else:
			count_0 += 1
			for j in range(len(arr[0])):
				countij_0[j] = countij_0[j] + arr[i][j]
	pij_0 = []
	pij_1 = []
	for i in range(50):
		pij_0.append(countij_0[i]/float(count_0))
		pij_1.append(countij_1[i]/float(count_1))

	return pij_0, pij_1




def convert_ip_arr(input):
	for i in range(len(input)):
		for j in range(len(input[0])):
			if input[i][j] > 0:
				input[i][j] = 1

	return input

def find_discriminant(class_prob, dimension_prob, input_row):
	class_val = math.log(class_prob)
	summation = 0.0
	for i in range(len(input_row)):
		summation = summation + input_row[i]*math.log(dimension_prob[i]) + (1-input_row[i])*math.log(1-dimension_prob[i])

	return class_val + summation


def main_indpendent_var(input_col, output_col):

	#input_col, output_col = util.read_file_data(TRAINING_DATA)
	pc_0, pc_1 = calculate_prio(output_col)
	binary_ip = convert_ip_arr(input_col)
	print(len(input_col), len(binary_ip), len(output_col))
	pij_0, pij_1 = calculate_probability_for_each_dimension(binary_ip, output_col)

	#validation_input = convert_ip_arr(input_col[4000:4500])
	pred = []
	count = 0
	for i in range(len(binary_ip)):
		c0 = find_discriminant(pc_0, pij_0, binary_ip[i])
		c1 = find_discriminant(pc_1, pij_1, binary_ip[i])
		if c0 < c1:
			pred.append(1)
			if output_col[i] == 1:
				count += 1
		else:
			pred.append(0)
			if output_col[i] == 0:
				count += 1
	if ENABLE_LOG == True:
		print("Count", count)

	return pred, pc_0, pc_1, pij_0, pij_1

#Data should be in Row Major fashion
def test(pc_0, pc_1, pij_0, pij_1, data_in):
	binary_ip = convert_ip_arr(data_in)
	pred = []
	for i in range(len(binary_ip)):
		c0 = find_discriminant(pc_0, pij_0, binary_ip[i])
		c1 = find_discriminant(pc_1, pij_1, binary_ip[i])
		if c0 < c1:
			pred.append(1)
		else:
			pred.append(0)

	return pred

if __name__ == '__main__':
	input_col, output_col = util.read_file_data_rowmajor(TRAINING_DATA)
	_ = main_indpendent_var(input_col[:4000], output_col[:4000])