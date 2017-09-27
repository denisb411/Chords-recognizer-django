from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError


class BestSKLearnModel(object):

	def __init__(self):
		self.__modelLogisticRegression = LogisticRegression(C=1.0, class_weight=None,dual=False, 
									fit_intercept=True, intercept_scaling=1, 
									penalty='l2', random_state=None, tol=0.0001)

		self.__state = 'NOT_TRAINED'

		self.__modelOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
		self.__modelOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
		self.__modelMultinomial = MultinomialNB()
		self.__modelDecisionTreeClassifier = DecisionTreeClassifier()
		self.__modelKNeighborsClassifier = KNeighborsClassifier()
		self.__modelLinearDiscriminantAnalysis = LinearDiscriminantAnalysis()
		self.__modelGaussianNB = GaussianNB()
		self.__modelMLPClassifier = MLPClassifier()
		self.__modelSGDClassifier = SGDClassifier()
		self.__modelSVC = SVC()

		self.__best_model = None		

	def __advance_state(self):
		if self.__state == 'NOT_TRAINED':
			self.__state = 'TRAINING'
		elif self.__state == 'TRAINING':
			self.__state = 'TRAINED'	

	def __check_state(self):
		if self.__state == 'NOT_TRAINED':
			raise NotFittedError('Classifier not trained yet')
		elif self.__state == 'TRAINING':
			raise NotFittedError('Model is in training process')	

	def fit(self, X_train, y_train):

		self.__advance_state()
		self.__scores = {}
		
		self.__scores.append(self.__fit_and_predict(self.__modelLogisticRegression, 
													X_train, y_train))

		self.__scores.append(self.__fit_and_predict(self.__modelOneVsOne, 
													X_train, y_train))

		self.__scores.append(self.__fit_and_predict(self.__modelMultinomial, 
													X_train, y_train))

		self.__scores.append(self.__fit_and_predict(self.__modelDecisionTreeClassifier, 
													X_train, y_train))

		self.__scores.append(self.__fit_and_predict(self.__modelKNeighborsClassifier, 
													X_train, y_train))

		self.__scores.append(self.__fit_and_predict(self.__modelLinearDiscriminantAnalysis, 
													X_train, y_train))

		self.__scores.append(self.__fit_and_predict(self.__modelGaussianNB, 
													X_train, y_train))

		self.__scores.append(self.__fit_and_predict(self.__modelMLPClassifier, 
													X_train, y_train))

		self.__scores.append(self.__fit_and_predict(self.__modelSGDClassifier, 
													X_train, y_train))

		self.__scores.append(self.__fit_and_predict(self.__modelSVC, 
													X_train, y_train))

		best_accuracy = 0
		for score in self.__scores:
			accuracy = score['accuracy']
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				self.__best_model = score['model']
		self.__best_model.fit()
		self.__advance_state()

	@property
	class scores(self):
		self.__check_state()
		return self.__scores

	def __fit_and_predict(self, model, X_train, y_train)
		k = 5
		#Evaluate a score by cross-validation
		scores = cross_val_score(model, X_train, y_train, cv = k, n_jobs = 4)
		accuracy = np.mean(scores) #calculates the average
		result['model'] = model
		result['accuracy'] = accuracy

		return result

	def predict(self, X):
		self.__check_state()
		return self.__best_model.predict(X)

    def accuracy_score(self, X_test, y_test):
    	y_pred = self.__best_model.predict(X_test)
		return accuracy_score(y_test, y_pred)