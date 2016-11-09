import sklearn as sk 
from sklearn import svm
from sklearn import decomposition
import sklearn.ensemble
import numpy as np
import util
from util import TRAINING_DATA

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer


def run_pca(features):
	pca = sk.decomposition.PCA()
	pca.fit(features)

	return pca


def knn_classifier_train(features, output):
	clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform', algorithm='kd_tree')
	clf.fit(features, output)

	return clf

def validate(name, clf, features, output):
	predict = clf.predict(features)
	accuracy = np.sum(predict[i] == output[i] for i in range(len(output))) /float(len(output))
	print('{0} accuracy= {1}'.format(name, accuracy))

def svm_classifier_train(features, output):
	svm = sk.svm.SVC(C=0.4, kernel='linear', random_state=20161101)
	svm.fit(features, output)

	return svm

def rf_classifier_train(features, output):
	rf=sk.ensemble.RandomForestClassifier(n_estimators=1000, random_state=20161101)
	rf.fit(features, output)

	return rf

def ada_boost_classifier(features, output):
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
	bdt.fit(features, output)

	return bdt

def mlp_classifier(features, output):
	clf = MLPClassifier(solver='lbfgs', alpha=1e-4,
		hidden_layer_sizes=(5, 1), random_state=20161102)
	clf.fit(features, output)

	return clf

def logisitic_classifier(features, output):
	lr = LogisticRegression()
	lr.fit(features, output)

	return lr

def main_():
	nd_features, output = util.read_file_data_rowmajor(TRAINING_DATA)
	nd_train = []
	nd_validate = []
	for i in range(len(nd_features)):
		nd_train.append(nd_features[i][:4000])
		nd_validate.append(nd_features[i][4000:5000])
	train_output = output[:4000]
	validate_output = output[4000:5000]


	'''
	pca = run_pca(np.array(nd_features[:4000]))
	pca_train_data = pca.transform(np.array(nd_features[:4000]))
	pca_validate_data = pca.transform( np.array(nd_features[4000:5000]))
	knn_pca = knn_classifier_train(pca_train_data, np.array(train_output))
	validate('knn_pca', knn_pca, pca_validate_data, np.array(validate_output))

	knn = knn_classifier_train(np.array(nd_features[:4000]), np.array(train_output))
	validate('knn', knn, np.array(nd_features[4000:5000]), np.array(validate_output))
	validate('Training error knn', knn, np.array(nd_features[:4000]), np.array(train_output))
	svm_pca = svm_classifier_train(pca_train_data, np.array(train_output))
	print(pca_train_data.shape)
	validate('svm_pca', svm_pca, pca_validate_data, np.array(validate_output))

	svm = svm_classifier_train(np.array(nd_features[:4000]), np.array(train_output))
	#svm_validate(svm, np.array(nd_features[4000:5000]), np.array(validate_output))
	validate('svm', svm, np.array(nd_features[4000:5000]), np.array(validate_output))
	validate('Training error svm', svm, np.array(nd_features[:4000]), np.array(train_output))
	rf_pca = svm_classifier_train(pca_train_data, np.array(train_output))
	validate('rf_pca', rf_pca, pca_validate_data, np.array(validate_output))
	rf = rf_classifier_train(np.array(nd_features[:4000]), np.array(train_output))
	validate('rf', rf, np.array(nd_features[4000:5000]), np.array(validate_output))
	validate('Training error rf', rf, np.array(nd_features[:4000]), np.array(train_output))

	bdt = ada_boost_classifier(np.array(nd_features[:4000]), np.array(train_output))
	validate('bdt', bdt, np.array(nd_features[4000:5000]), np.array(validate_output))
	validate('Training error bdt', bdt, np.array(nd_features[:4000]), np.array(train_output))

	mlp = mlp_classifier(np.array(nd_features[:4000]), np.array(train_output))
	validate('mlp', mlp, np.array(nd_features[4000:5000]), np.array(validate_output))
	validate('Training error mlp:', mlp, np.array(nd_features[:4000]), np.array(train_output))
	'''
	tfidf_transformer = TfidfTransformer()
	X_train_tfidf = tfidf_transformer.fit_transform(nd_features)
	lr = logisitic_classifier(X_train_tfidf[:4000], np.array(train_output))
	validate('lr', lr, X_train_tfidf[4000:5000], np.array(validate_output))
	validate('Training error lr:', lr, np.array(nd_features[:4000]), np.array(train_output))

	from sklearn.naive_bayes import MultinomialNB
	from sklearn.pipeline import Pipeline
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.grid_search import GridSearchCV

	text_clf = Pipeline([
		('tfidf', TfidfTransformer()),
		('clf', MultinomialNB()),
		])
	clf = text_clf.fit(X_train_tfidf[:4000], np.array(train_output))
	validate('clf', clf, X_train_tfidf[4000:5000], np.array(validate_output))

	parameters = {
	'vect__max_n': (1, 2),
	'tfidf__use_idf': (True, False),
	'clf__alpha': (1e-2, 1e-3),
	}

	gs_clf = GridSearchCV(text_clf, parameters, n_jobs=1)
	gs_clf = gs_clf.fit(np.array(nd_features[:4000]), np.array(train_output))
	validate('gs_clf', gs_clf, np.array(nd_features[4000:5000]), np.array(validate_output))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.decomposition import SparsePCA

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def benchmark(clf, X_train, y_train, X_test, y_test):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print()

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

from scipy.sparse import csr_matrix
from sklearn import preprocessing
def main__():

	nd_features, output = util.read_file_data_rowmajor(TRAINING_DATA)
	nd_train = []
	nd_validate = []
	results = []
	print(np.sum(output)*100.0/float(len(output)))
	scaler = preprocessing.MaxAbsScaler().fit(nd_features)
	X_train = csr_matrix(np.array(nd_features[:4000]))
	y_train = np.array(output[:4000])
	X_test = csr_matrix(np.array(nd_features[4000:5000]))
	Y_test = np.array(output[4000:5000])

	print('=' * 80)
	print("Naive Bayes")
	for clf, name in ( (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"), (Perceptron(n_iter=50), "Perceptron"), (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"), (KNeighborsClassifier(n_neighbors=10), "kNN"), (RandomForestClassifier(n_estimators=100), "Random forest"), (LogisticRegression(), "Logistic Regression")):
		print('=' * 80)
		print(name)
		results.append(benchmark(clf, X_train, y_train, X_test, Y_test))

	# Train SGD with Elastic Net penalty
	print('=' * 80)
	print("Elastic-Net penalty")
	results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), X_train, y_train, X_test, Y_test))

	# Train NearestCentroid without threshold
	print('=' * 80)
	print("NearestCentroid (aka Rocchio classifier)")
	results.append(benchmark(NearestCentroid(), X_train, y_train, X_test, Y_test))
	# Train sparse Naive Bayes classifiers

	print('=' * 80)
	print("Naive Bayes")
	results.append(benchmark(MultinomialNB(), X_train, y_train, X_test, Y_test))
	results.append(benchmark(BernoulliNB(alpha=.01), X_train, y_train, X_test, Y_test))

	print('=' * 80)
	print("LinearSVC with L1-based feature selection")
	# The smaller C, the stronger the regularization.
	# The more regularization, the more sparsity.
	results.append(benchmark(Pipeline([('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),('classification', LinearSVC())]), X_train, y_train, X_test, Y_test))

def combine_two_features(nd_features, output):
	added_features = []
	added_features_pos_i=[]
	added_features_pos_j=[]
	for i in range(0, 49):
		for j in range(i+1, 50):
			c_i_j_0 = 0
			c_i_j_1 = 0
			comb_arr = []
			for k in range(len(nd_features[0])):
				if nd_features[i][k] > 0 and nd_features[j][k] > 0:
					if output[k] == 1:
						c_i_j_1 += 1
					else:
						c_i_j_0 += 1
					comb_arr.append(1)
				else:
					comb_arr.append(0)
			if c_i_j_1 == 0:
				c_i_j_1 = 1
			if c_i_j_0 == 0:
				c_i_j_0 = 1
			if c_i_j_0 > c_i_j_1:
				ratio = c_i_j_0 / float(c_i_j_1)
			else:
				ratio = c_i_j_1 / float(c_i_j_1)
			if (c_i_j_0 < 20 or c_i_j_1 < 20) and (c_i_j_0 + c_i_j_1) > 10:
			#if ratio > 2:
				print(c_i_j_0, c_i_j_1, i, j)
				added_features.append(comb_arr)
				added_features_pos_i.append(i)
				added_features_pos_j.append(j)
	added_nd_features = []
	for i in range(len(nd_features)):
		added_nd_features.append(nd_features[i])
	for i in range(len(added_features)):
		added_nd_features.append(added_features[i])

	return added_nd_features, added_features_pos_i, added_features_pos_j

def convert_to_added_nd_features(input_data, added_features_pos_i, added_features_pos_j):
	for i in range(len(added_features_pos_i)):
		column = []
		for j in range(len(input_data[0])):
			if input_data[added_features_pos_i[i]][j] > 0 and input_data[added_features_pos_j[i]][j] > 0:
				column.append(1)
			else:
				column.append(0)
		input_data.append(column)

	return input_data



def main():
	#Just use 4000 data for the study
	file_header = util.read_file_header(TRAINING_DATA)
	nd_features_, output = util.read_file_data(TRAINING_DATA)
	nd_features = []
	validate_in = []
	validate_out = []
	for i in range(len(nd_features_)):
		nd_features.append(nd_features_[i][:4000])
		validate_in.append(nd_features_[i][4000:5000])
	validate_out.append(output[4000:5000])
	count_0_is_0 = []
	count_0_is_1 = []
	count_1_is_0 = []
	count_1_is_1 = []
	count_0 = []
	count_1 = []
	X_train, added_features_pos_i, added_features_pos_j = combine_two_features(nd_features, output[:4000])
	X_train = np.array(X_train).T
	y_train = output[:4000]
	'''
	X_test= X_train
	Y_test = y_train
	'''
	X_test = convert_to_added_nd_features(validate_in, added_features_pos_i, added_features_pos_j)
	X_test = np.array(X_test).T
	Y_test = output[4000:]

	results = []
	for clf, name in ( (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"), (Perceptron(n_iter=50), "Perceptron"), (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"), (KNeighborsClassifier(n_neighbors=10), "kNN"), (RandomForestClassifier(n_estimators=100), "Random forest"), (LogisticRegression(), "Logistic Regression")):
		print('=' * 80)
		print(name)
		results.append(benchmark(clf, X_train, y_train, X_test, Y_test))


	# Train SGD with Elastic Net penalty
	print('=' * 80)
	print("Elastic-Net penalty")
	results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"), X_train, y_train, X_test, Y_test))

	# Train NearestCentroid without threshold
	print('=' * 80)
	print("NearestCentroid (aka Rocchio classifier)")
	results.append(benchmark(NearestCentroid(), X_train, y_train, X_test, Y_test))
	# Train sparse Naive Bayes classifiers

	print('=' * 80)
	print("Naive Bayes")
	results.append(benchmark(MultinomialNB(), X_train, y_train, X_test, Y_test))
	results.append(benchmark(BernoulliNB(alpha=.01), X_train, y_train, X_test, Y_test))

	print('=' * 80)
	print("LinearSVC with L1-based feature selection")
	# The smaller C, the stronger the regularization.
	# The more regularization, the more sparsity.
	results.append(benchmark(Pipeline([('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),('classification', LinearSVC())]), X_train, y_train, X_test, Y_test))
	'''
	benchmark(MultinomialNB(), X_train, y_train, X_test, Y_test)
	benchmark(MultinomialNB(), X_train, y_train, X_train, y_train)
	benchmark(LogisticRegression(), X_train, y_train, X_test, Y_test)
	benchmark(LogisticRegression(), X_train, y_train, X_train, y_train)
	'''


	'''
	c_1 = np.sum(output)
	c_0 = len(output) - c_1

	fract_0 = c_1/float(c_0+c_1)
	fract_1 = c_0/float(c_0+c_1)
	goodness_list = []
	print(fract_0, fract_1)
	feature_combination = []
	combination_value = []
	c_i_j_0 = 0
	c_i_j_1 = 0
	max_corr = -100

	for i in range(0, 48):
		for j in range(i+1, 49):
			c_i_j_0 = 0
			c_i_j_1 = 0
			val = np.corrcoef(nd_features[i], nd_features[j])
			if val[0][1] > max_corr:
				i_pos = i 
				j_pos = j
				max_corr = val[0][1]
	print("max_corr", max_corr, file_header[i_pos], file_header[j_pos])
			for k in range(len(nd_features[0])):
				if nd_features[i][k] > 0 and nd_features[j][k] > 0:
					if output[k] == 1:
						c_i_j_1 += 1
					else:
						c_i_j_0 += 1
			c_i_j_0 = c_i_j_0*fract_0
			c_i_j_1 = c_i_j_1*fract_1
			if ((c_i_j_0/c_i_j_1) > 5) or ((c_i_j_1 / c_i_j_0) > 5):
				print(c_i_j_0, c_i_j_1, file_header[i], file_header[j])
	for i in range(0, 48):
		for j in range(i+1, 49):
			for l in range( j+1, 50):
				c_i_j_0 = 0
				c_i_j_1 = 0
				for k in range(len(nd_features[0])):
					if nd_features[i][k] > 0 and nd_features[j][k] > 0 and nd_features[l][k] > 0:
						if output[k] == 1:
							c_i_j_1 += 1
						else:
							c_i_j_0 += 1
				if (c_i_j_0 < 5 or c_i_j_1 < 5) and (c_i_j_0 + c_i_j_1) > 32:
					print(c_i_j_0, c_i_j_1, i, j, l)
				#feature_combination.append((i, j))
				#combination_value.append((c_i_j_0, c_i_j_1))
				#if (c_i_j_0 + c_i_j_1) > 50:
				#	print(c_i_j_0, c_i_j_1)
	#print(feature_combination)
	print(combination_value)
	'''
	'''





	for i in range(len(nd_features)):
		c_0_is_0 = 0
		c_0_is_1 = 0
		c_1_is_0 = 0
		c_1_is_1 = 0
		goodness_val = 0
		for j in range(len(nd_features[0])):
			if nd_features[i][j] > 0:
				if output[j] == 1:
					c_1_is_1 += 1
					goodness_val += fract_1
				else:
					c_0_is_1 += 1
					goodness_val -= fract_0
		#count_0_is_0.append(c_0_is_0/float(c_0))
		#if c_0_is_1/float(c_0) > 1.0:
		print(c_0_is_1, c_0, c_1, c_1_is_1, goodness_val)
		goodness_list.append(goodness_val)
		count_0_is_1.append(c_0_is_1/float(c_0_is_1 + c_1_is_1))
		count_1_is_1.append(c_1_is_1/float(c_0_is_1 + c_1_is_1))
	sum_0 = np.sum(count_0_is_1)
	sum_1 = np.sum(count_1_is_1)
	count_0_is_1 = [count_0_is_1[i]/sum_0 for i in range(len(count_0_is_1))]
	count_1_is_1 = [count_1_is_1[i]/sum_1 for i in range(len(count_1_is_1))]
	print("0", c_0)
	print("1", c_1)
	print("0 is 0", count_0_is_0)
	print("")
	print("")
	print("0 is 1", count_0_is_1)
	print("")
	print("")
	print("1 is 0", count_1_is_0)
	print("")
	print("")
	print("1 is 1", count_1_is_1)

	nd_features, output = util.read_file_data_rowmajor(TRAINING_DATA)
	validate_in = nd_features[4000:5000]
	#output = output[4000:5000]
	correct_count = 0
	for i in range(0, 4000):
		count_prob_0 = 0
		count_prob_1 = 0
		for j in range(len(nd_features[i])):
			#if validate_in[i][j] == 0:
				#if output[j] == 0:
					#count_prob_0 += count_0_is_0[j]

				#else:
				#count_prob_1 += count_0_is_1[j]
			#else:
			import math
			if nd_features[i][j] > 0:
				if goodness_list[j] < 0:
					count_prob_0 += math.fabs(goodness_list[j]*(count_0_is_1[j]))
				if goodness_list[j] > 0:
					count_prob_1 += math.fabs(goodness_list[j]*(count_1_is_1[j]))

		print(output[i], fract_1*count_prob_0, fract_0*count_prob_1)
		if count_prob_0 > count_prob_1:
			if output[i] == 0:
				correct_count += 1
		else:
			if output[i] == 1:
				correct_count += 1

	print(correct_count, np.sum(output[:4000]))
	'''



if __name__ == '__main__':
	main()