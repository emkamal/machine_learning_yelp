import random
import numpy as np

TEST_DATA = "ex2_testdata.txt"
TRAINING_DATA = "ex2_traindata.txt"

def create_input_matrix(dim, dataList):
	x = np.array([])
	x.resize(len(dataList), dim + 1)
	for i in range(len(dataList)):
		x.itemset((i, 0), 1)
		val = 1.0
		for j in range(dim):
			val *= dataList[i]
			x.itemset((i, j+1), val)

	return x


def create_output_matrix(dataList):
	y = np.array(([]))
	y.resize(1, len(dataList))
	for i in range(num):
		y[i] = dataList[i]

	return y.transpose()


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


def find_square_error(dim, w, validationData_in, validationData_out):
	sum_of_error = 0.0
	for i in range(len(validationData_in)):
		error_val = w[0][0]
		val = 1.0
		for j in range(dim):
			val = val * validationData_in[i]
			error_val = error_val + (val*w[j+1][0])
		sum_of_error = sum_of_error + ((error_val-validationData_out[i])*(error_val-validationData_out[i]))
	return sum_of_error/len(validationData_in)

			


def read_file_data(file_name):
	column_1 = []
	column_2 = []
	file = open(file_name, "r")
	for line in file:
		strData = line.split();
		column_1.append(float(strData[0]))
		column_2.append(float(strData[1]))
	file.close()
	return column_1, column_2

def seperate_train_validate_data(iteration, data_in, data_out ):
	trainData_in = []
	trainData_out = []
	validationData_in = []
	validationData_out = []
	for i in range(5):
		validationData_in.append( data_in[5*iteration+i] )
		validationData_out.append( data_out[5*iteration+i] )

	for i in range(len(data_in)):
		if(not ((i>=(iteration*5)) and (i < (iteration*5+5)))):
			trainData_in.append(data_in[i])	
			trainData_out.append(data_out[i])

	return trainData_in, trainData_out, validationData_in, validationData_out

def run_linear_regression(dim):
	duration_input, rate_output = read_file_data(TRAINING_DATA)
	avg_w = []
	avg_error = 0
	for i in range(dim+1):
		avg_w.append(0.0)
	for i in range(10):
		trainData_in, trainData_out, validationData_in, validationData_out = seperate_train_validate_data(i, duration_input, rate_output)
		input_array = create_input_matrix(dim, trainData_in)
		output_array = create_output_matrix(trainData_out)
		w = find_w(input_array, output_array)
		for i in range(dim+1):
			avg_w[i] += w[i][0]
		avg_error += find_square_error( dim, w, validationData_in, validationData_out )
	for i in range(dim+1):
		avg_w[i] = avg_w[i]/10.0
	#print(avg_error)
	#print(avg_w, len(avg_w))
	#print(avg_w)
	testData_in, testData_out = read_file_data(TEST_DATA)
	avg_error = find_square_error( dim, avg_w, testData_in, testData_out )
	print(avg_error)

	import matplotlib.pyplot as plt
	approximate_output_array = []
	for i in range(len(testData_in)):
		approximate_output_array.append(0.0)
	for i in range(len(testData_in)):
		val = 1.0
		out_val = avg_w[0]
		for j in range(dim):
			val = val * testData_in[i]
			out_val = out_val + avg_w[j+1]*val
		approximate_output_array[i] = float(out_val)

	plt.scatter(testData_in, testData_out)
	#plt.plot(testData_in, testData_out)
	plt.scatter(testData_in, approximate_output_array,c='r')
	plt.show()



def run_algorithms():
	print("Output from Generic Algorithm.")
	print("")
	print("")
	print("")

	run_linear_regression(10)
	'''
	for i in range(1):
		print("Dim="+str(i+1))
		run_linear_regression(i+1)
		'''


def test_square_error():
	w = [[5],[2]]
	testData_in = [3, 4, 5]
	testData_out = [6, 8, 10]
	val = find_square_error_1d( w[0][0], w[1][0], testData_in, testData_out)
	avg_error = find_square_error( 1, w, testData_in, testData_out )
	print( val, avg_error )

if __name__ == "__main__":
	#test_square_error()
	run_algorithms()
