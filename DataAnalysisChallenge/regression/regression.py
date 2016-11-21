import numpy as np
import statsmodels.api as sm

from numpy import genfromtxt

training_data = genfromtxt('regression_dataset_training.csv', delimiter=',')
training_data = np.delete(training_data, (0), axis=0)

r = training_data[:, 51]
x = training_data[:, 1:51]
x1 = x[:, 0]

poly_deg = 10;
n = 5000;

perm = np.random.permutation(n)

basis_funs = np.ones((n, poly_deg+1))

for i in range(0, poly_deg):
    basis_funs[:, (i+1)] = pow(x1[:], i)

valid_preds = np.zeros((n, poly_deg+1))
train_preds = np.zeros((n, poly_deg+1))

for i in range(0, n):
    

#y = [1,2,3,4,3,4,5,4,5,5,4,5,4,5,4,5,6,5,4,5,4,3,4]
#
#x = [
#     [4,2,3,4,5,4,5,6,7,4,8,9,8,8,6,6,5,5,5,5,5,5,5],
#     [4,1,2,3,4,5,6,7,5,8,7,8,7,8,7,8,7,7,7,7,7,6,5],
#     [4,1,2,5,6,7,8,9,7,8,7,8,7,7,7,7,7,7,6,6,4,4,4]
#     ]

#def reg_m(y, x):
#    ones = np.ones(len(x[0]))
#    X = sm.add_constant(np.column_stack((x[0], ones)))
#    for ele in x[1:]:
#        X = sm.add_constant(np.column_stack((ele, X)))
#    results = sm.OLS(y, X).fit()
#    return results

#if __name__ == '__main__':
#	print(reg_m(y, x).summary())