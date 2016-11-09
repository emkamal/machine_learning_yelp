import random
import numpy as np
import util
import math
from util import COL_LEN, ENABLE_LOG, TRAINING_DATA, TESTING_DATA, TESTING_DATA_SOLUTION


#Left will point towrads the one with more zeros if present
#Right will point towards the one with more ones if present
class tree_node:
	def __init__(self, count_0, count_1, column_index):
		self.column_index = column_index
		self.count_0 = count_0
		self.count_1 = count_1
		self.left = None
		self.right = None

def calculate_entropy(output):
	dm_0 = 0
	dm_1 = 0
	calc_ok = True
	for i in range(len(output)):
		if output[i] == 0:
			dm_0 += 1
		else:
			dm_1 += 1

	dm = dm_0 + dm_1
	if dm == 0:
		dm = 1
		calc_ok = False
		if ENABLE_LOG == True:
			print("Divide by zero")
			print(dm, dm_0, dm_1, dm_0/dm, dm_1/dm)

	if dm_0 == 0:
		dm_0 = dm
		calc_ok = False
		if ENABLE_LOG == True:
			print("dm_0 is zero", dm_0)
			print(dm, dm_0, dm_1, dm_0/dm, dm_1/dm)

	if dm_1 == 0:
		dm_1 = dm
		calc_ok = False
		if ENABLE_LOG == True:
			print("dm_1 is zero")
			print(dm, dm_0, dm_1, dm_0/dm, dm_1/dm)

	#print(dm, dm_0, dm_1, dm_0/dm, dm_1/dm)
	entropy = -(dm_1 *(math.log(dm_1/dm))/dm) -( dm_0* (math.log(dm_0/dm))/dm )
	return entropy, calc_ok

def gain(input, output, col):
	m_0_out = []
	m_1_out = []
	count_1 = 0
	count_0 = 0

	for i in range(len(output)): #Suppose the reading is done in row major!!!!
		if input[i][col] > 0:
			m_1_out.append(output[i])
			count_1 += 1
		else:
			m_0_out.append(output[i])
			count_0 += 1
	hd_m, calc = calculate_entropy(output)
	hd_m_0, calc = calculate_entropy(m_0_out)
	hd_m_1, calc = calculate_entropy(m_1_out)
	gain = hd_m - ( count_0/len(output)*hd_m_0 + count_1/len(output)*hd_m_1)
	return gain


def split_tree(input, output, column_avail):
	isSplit = False
	list_gain = []
	for i in range(len(column_avail)):
		if column_avail[i] == 0:
			list_gain.append(-1000)
			continue
		list_gain.append(gain(input, output, i))

	max_gain = -1000
	max_gain_pos = -1
	for i in range(len(list_gain)):
		if list_gain[i] > max_gain:
			max_gain = list_gain[i]
			max_gain_pos = i
	if max_gain < 0.01:
		return isSplit, None, None, None, None, len(output)-np.sum(output), np.sum(output), None
	isSplit = True
	input_0 = []
	input_1 = []
	output_0 = []
	output_1 = []
	column_avail[max_gain_pos] = 0
	count_0  = 0
	count_1 = 0
	for i in range(len(input)):
		if input[i][max_gain_pos] > 0:
			input_1.append(input[i])
			output_1.append(output[i])
		else:
			input_0.append(input[i])
			output_0.append(output[i])
		if output[i] == 0:
			count_0 += 1
		else:
			count_1 += 1

	return isSplit, input_0, input_1, output_0, output_1, count_0, count_1, max_gain_pos


def build_tree(input, output, col_avail, depth):

	if depth < 0:
		return None
	column_avail = []
	for i in range(len(col_avail)):
		column_avail.append(col_avail[i])
	tree_split, input_l, input_r, output_l, output_r, c0, c1, col = split_tree(input, output, column_avail)
	
	node = tree_node(c0, c1, col)
	if tree_split == False:
		return node
	if len(output_l) < 3 or len(output_r) < 3:
		return node
	node.left = build_tree(input_l, output_l, column_avail, depth - 1)
	node.right = build_tree(input_r, output_r, column_avail, depth -1)
	
	return node

def validate_decision_tree(node, input):
	if node.left == None and node.right == None:
		if node.count_1 > node.count_0:
			return 1
		else:
			return 0
	if input[node.column_index] == 0:
		return validate_decision_tree(node.left, input)	
	else:
		return validate_decision_tree(node.right, input)


def tree_traversal(node):
	if node == None:
		return
	print(node.column_index, node.count_0, node.count_1)
	tree_traversal(node.left)
	tree_traversal(node.right)

def validate(node, input, output):
	if node == None:
		print("Something went terribly wrong")
		return
	#print("Tree start")
	#tree_traversal(node)
	correct_predict = 0
	#print("Start:")
	pred = []
	for i in range(len(input)):
		val = validate_decision_tree(node, input[i])
		pred.append(val)
		if val == output[i]:
			correct_predict += 1
	if ENABLE_LOG == True:
		print(correct_predict)

	return pred

#Use all the data to train
def main(input_col, output_col):
	column_avail = []
	for i in range(len(input_col[0])):
		column_avail.append(1)
	if ENABLE_LOG == True:
		print(len(column_avail))
	node = build_tree(input_col, output_col, column_avail, 20)
	#input, output = util.read_file_data_rowmajor(TRAINING_DATA)
	if ENABLE_LOG == True:
		print("Training Error:")
	pred = validate(node, input_col, output_col)
	return pred, node

def test(node, data_in):
	if node == None:
		print("Something went Wrong. Tree is Empty.")
		return None

	#print("Start:")
	pred = []
	for i in range(len(data_in)):
		val = validate_decision_tree(node, data_in[i])
		pred.append(val)

	return pred

if __name__ == '__main__':
	input_col, output_col = util.read_file_data_rowmajor(TRAINING_DATA)
	_, node = main(input_col, output_col)
	pred = test(node, input_col)
	count = 0
	for i in range(len(pred)):
		if pred[i] == output_col[i]:
			count += 1
	print("Training Error", count, np.sum(output_col), util.logloss(output_col, pred))
	input_col, output_col = util.read_file_data_rowmajor_test(TESTING_DATA, TESTING_DATA_SOLUTION)
	pred = test(node, input_col)
	count = 0
	for i in range(len(pred)):
		if pred[i] == output_col[i]:
			count += 1
	print("Training Error", count, np.sum(output_col), util.logloss(output_col, pred))

