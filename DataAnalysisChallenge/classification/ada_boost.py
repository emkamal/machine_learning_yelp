import numpy as np
import feature_selection as fs
import decision_trees as dt
import Multivariate_binomial as mb
import util
from  util import TRAINING_DATA

def add(input, input_col):
	for i in range(len(input_col)):
		input[i].append(input_col[i])

def main():
	input_col, output_col = util.read_file_data(TRAINING_DATA)

	input_col_row, output_col_row = util.read_file_data_rowmajor(TRAINING_DATA)

	train_list_in = []
	for i in range(len(input_col)):
		train_list_in.append(input_col[i][:4000])

	#Bad programming practice !!!! Should have thought it through before.
	#Decision Trees and Multivariate Binomial uses the matrix as row of input data ie 5000x50
	#Linear Regression uses it as row of columns ie 50x5000
	#Need to adapt accordingly
	#We just need output_col for verification
	print("Decision Trees")
	pred1, node = dt.main(input_col_row[:4000], output_col_row[:4000])
	'''
	input_list2 = []
	output_list2 = []
	count = 0
	#For each mistake row also add one correct row so that the next iteration will not only be about the wrong data
	wrong_count = 0
	for i in range(len(pred1)):
		if pred1[i] != output_col_row[i]:
			input_list2.append(input_col_row[i])
			output_list2.append(output_col_row[i])
			wrong_count += 1
		else:
			count += 1
			if wrong_count > 0:
				#input_list2.append(input_col_row[i])
				#output_list2.append(output_col_row[i])
				wrong_count -= 1
	print("Count: ", count, " Length: ", len(input_list2), 4000 - count, "pred_len", len(pred1))

	'''
	print("Multivariate Binomial")
	pred2, pc_0, pc_1, pij_0, pij_1 = mb.main_indpendent_var(input_col_row[:4000], output_col_row[:4000])
	'''
	count = 0
	wrong_count = 0
	input_list3_col = []
	for i in range(50):
		list1 = []
		input_list3_col.append(list1)
	output_list3 = []
	for i in range(len(pred2)):
		if pred2[i] != output_list2[i]:
			add( input_list3_col, input_col_row[i] )
			output_list3.append(output_col_row[i])
			wrong_count += 1
		else:
			count += 1
			if wrong_count > 0:
				#add( input_list3_col, input_col_row[i] )
				#output_list3.append(output_col_row[i])
				wrong_count -= 1
	print("Count: ", count, "Pred_len", len(pred2), "Origin Len=", len(output_list3), "in len:", len(input_list3_col))
	'''
	print("Linear Regression with Feature Selection")
	pred3, param_list, selected_var = fs.run_linear_regression(train_list_in, output_col[:4000])
	'''
	count = 0
	for i in range(len(pred3)):
		if pred3[i] != output_list3[i]:
			count += 1
	print("Count: ", count, "Pred_len", len(pred3), "Origin Len=", len(input_list3_col))
	'''
	count_1 = 0
	count_2 = 0
	count_3 = 0
	for i in range(4000):
		if pred1[i] == output_col[i]:
			count_1 += 1
		if pred2[i] == output_col[i]:
			count_2 += 1
		if pred3[i] == output_col[i]:
			count_3 += 1
	w1 = count_1/4000.0
	w2 = count_2/4000.0
	w3 = count_3/4000.0
	w1_norm = w1 / (w1+w2+w3)
	w2_norm = w2 / (w1+w2+w3)
	w3_norm = w3 / (w1+w2+w3)


	test_list_in = []
	for i in range(len(input_col)):
		test_list_in.append(input_col[i][4000:])

	print("Testing:")
	print("Testing:")
	print("Testing:")
	pred_validate_1 = dt.test(node, input_col_row[4000:])
	pred_validate_2 = mb.test(pc_0, pc_1, pij_0, pij_1, input_col_row[4000:])
	pred_validate_3 = fs.test(param_list, selected_var, test_list_in)
	count = 0
	ncount = 0
	for i in range(len(pred_validate_3)):
		weight_zero = 0.0
		weight_one = 0.0
		if pred_validate_1[i] == 0:
			weight_zero += w1_norm
		else:
			weight_one += w1_norm
		if pred_validate_2[i] == 0:
			weight_zero += w2_norm
		else:
			weight_one += w2_norm
		if pred_validate_3[i] == 0:
			weight_zero += w3_norm
		else:
			weight_one += w3_norm
		if weight_one > weight_zero:
			if output_col[i+4000] == 1:
				count += 1
			else:
				ncount += 1
		else:
			if output_col[i+4000] == 0:
				count += 0
			else:
				ncount += 1

	print("Test data", len(pred_validate_1), count, ncount)


if __name__ == '__main__':
	main()