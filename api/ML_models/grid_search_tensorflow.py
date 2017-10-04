from .tensorflow_models import CNNClassifier, DNNClassifier

class GridSearchCNN(object):
    def __init__(self, params, k_fold=5):
        """GridSearch on CNN with cross_validation with k_fold = 5"""
        self.n_hidden_layers = params['n_hidden_layers']
        self.n_neurons = params['n_neurons']
        self.optimizer_class = params['optimizer_class']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.activation = params['activation']
        self.dropout_rate = params['dropout_rate']
        self.conv1 = params['conv1']
        self.conv2 = params['conv2']
        self.architecture = params['architecture']
        self.best_params = None
        self.k_fold = k_fold

    def fit(self, X, y, X_valid=None, y_valid=None):

        scores = []
        for n_hidden_layers in self.n_hidden_layers:
            for n_neurons in self.n_neurons:
                for optimizer_class in self.optimizer_class:
                    for learning_rate in self.learning_rate:
                        for batch_size in self.batch_size:
                            for activation in self.activation:
                                for dropout_rate in self.dropout_rate:
                                    for conv1 in self.conv1:
                                        for conv2 in self.conv2:
                                            for architecture in self.architecture:
                                                accuracy_rate = 0
                                                folds = self.k_fold
                                                if folds <= 1:
                                                    print("Trining CNN with parameters: " + 
                                                            "n_hidden_layers: %d, " % (n_hidden_layers) +
                                                            "n_neurons: %d, " % (n_neurons) +
                                                            "optimizer_class: %s, " % (optimizer_class) +
                                                            "learning_rate: %d, " % (learning_rate) +
                                                            "batch_size: %d, " % (batch_size) +
                                                            "activation: %s, " % (activation) +
                                                            "dropout_rate: %d, " % (dropout_rate) +
                                                            "architecture: %d, " % (architecture) +
                                                            "conv1: %s, " % (conv1) +
                                                            "conv2: %s ." % (conv2)
                                                        )

                                                    trainRange = int(len(X) * 0.85)
                                                    testRange = int(len(X) * 0.15)

                                                    X_train = X[:trainRange]
                                                    y_train = y[:trainRange]

                                                    X_test = X[:testRange]
                                                    y_test = y[:testRange]

                                                    print("shapes:")
                                                    print(X_train.shape, X_test.shape)
                                                    print(y_train.shape, y_test.shape)

                                                    cnn = CNNClassifier(n_hidden_layers=n_hidden_layers, n_neurons=n_neurons, optimizer_class=optimizer_class,
                                                        learning_rate=learning_rate, activation=activation, conv1=conv1, conv2=conv2, architecture=architecture)

                                                    cnn.fit(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
                                                    accuracy_rate += cnn.accuracy_score(X_test, y_test)
                                                    del cnn

                                                else:
                                                    k_fold_samples = int(len(X) / folds)
                                                    print("Trining CNN with parameters: " + 
                                                            "n_hidden_layers: %d, " % (n_hidden_layers) +
                                                            "n_neurons: %d, " % (n_neurons) +
                                                            "optimizer_class: %s, " % (optimizer_class) +
                                                            "learning_rate: %d, " % (learning_rate) +
                                                            "batch_size: %d, " % (batch_size) +
                                                            "activation: %s, " % (activation) +
                                                            "dropout_rate: %d, " % (dropout_rate) +
                                                            "architecture: %d, " % (architecture) +
                                                            "conv1: %s, " % (conv1) +
                                                            "conv2: %s ." % (conv2)
                                                        )
                                                    for k in range(folds):
                                                        print("Training and testing fold %i" % k)
                                                        range1 = k_fold_samples * ((folds - 1) - k)
                                                        range2 = k_fold_samples * (folds - k)

                                                        print("ranges:")
                                                        print(range1, range2)

                                                        X_train = X[:range1]
                                                        X_train = np.vstack([X_train, X[range2:]])
                                                        y_train = y[:range1]
                                                        y_train = np.append(y_train, y[range2:])

                                                        X_test = X[range1:range2]
                                                        y_test = y[range1:range2]

                                                        print("shapes:")
                                                        print(X_train.shape, X_test.shape)
                                                        print(y_train.shape, y_test.shape)

                                                        cnn = CNNClassifier(n_hidden_layers=n_hidden_layers, n_neurons=n_neurons, optimizer_class=optimizer_class,
                                                            learning_rate=learning_rate, activation=activation, conv1=conv1, conv2=conv2, architecture=architecture)

                                                        cnn.fit(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
                                                        accuracy_rate += cnn.accuracy_score(X_test, y_test)
                                                        del cnn

                                                final_accuracy_rate = accuracy_rate / folds
                                                score = {'n_hidden_layers': n_hidden_layers,
                                                                'n_neurons' : n_neurons,
                                                                'optimizer_class' : optimizer_class,
                                                                'learning_rate' : learning_rate,
                                                                'batch_size' : batch_size,
                                                                'activation' : activation,
                                                                'dropout_rate' : dropout_rate,
                                                                'conv1' : conv1,
                                                                'conv2' : conv2,
                                                                'architecture' : architecture,
                                                                'accuracy_rate' : final_accuracy_rate,
                                                            }
                                                scores.append(score)
                                                print("******************************************")
                                                print(scores[-1])
                                                print("******************************************")

                                                with open("search_results/gridSearchDNN_results.txt","a") as f:
                                                    f.write(str(score) + "\n")

        best_score = 0
        for score in scores:
            accuracy = score['accuracy_rate']
            if accuracy > best_score:
                self.best_params = score
        print("Best parameters:")
        print(self.best_params)

class GridSearchDNN(object):
    def __init__(self, params, k_fold=5):
        """GridSearch on CNN with cross_validation with k_fold = 5"""
        self.n_hidden_layers = params['n_hidden_layers']
        self.n_neurons = params['n_neurons']
        self.optimizer_class = params['optimizer_class']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.activation = params['activation']
        self.dropout_rate = params['dropout_rate']
        self.best_params = None
        self.k_fold = k_fold

    def fit(self, X, y, X_valid=None, y_valid=None):

        scores = []
        for n_hidden_layers in self.n_hidden_layers:
            for n_neurons in self.n_neurons:
                for optimizer_class in self.optimizer_class:
                    for learning_rate in self.learning_rate:
                        for batch_size in self.batch_size:
                            for activation in self.activation:
                                for dropout_rate in self.dropout_rate:
                                    accuracy_rate = 0
                                    folds = self.k_fold
                                    k_fold_samples = int(len(X) / folds)
                                    for k in range(folds):
                                        print("Trining DNN with parameters: " + 
                                            "n_hidden_layers: %d, " % (n_hidden_layers) +
                                            "n_neurons: %d, " % (n_neurons) +
                                            "optimizer_class: %s, " % (optimizer_class) +
                                            "learning_rate: %d, " % (learning_rate) +
                                            "batch_size: %d, " % (batch_size) +
                                            "activation: %s, " % (activation) +
                                            "dropout_rate: %d, " % (dropout_rate)
                                        )
                                        print("Training and testing fold %i" % k)
                                        range1 = k_fold_samples * ((folds - 1) - k)
                                        range2 = k_fold_samples * (folds - k)

                                        print("ranges:")
                                        print(range1, range2)

                                        X_train = X[:range1]
                                        X_train = np.vstack([X_train, X[range2:]])
                                        y_train = y[:range1]
                                        y_train = np.append(y_train, y[range2:])

                                        X_test = X[range1:range2]
                                        y_test = y[range1:range2]

                                        print("shapes:")
                                        print(X_train.shape, X_test.shape)
                                        print(y_train.shape, y_test.shape)

                                        dnn = DNNClassifier(n_hidden_layers=n_hidden_layers, n_neurons=n_neurons, optimizer_class=optimizer_class,
                                            learning_rate=learning_rate, activation=activation)

                                        dnn.fit(X=X_train, y=y_train, X_valid=X_valid, y_valid=y_valid)
                                        accuracy_rate += dnn.accuracy_score(X_test, y_test)
                                        del dnn

                                    final_accuracy_rate = accuracy_rate / folds
                                    score = {'n_hidden_layers': n_hidden_layers,
                                                            'n_neurons' : n_neurons,
                                                            'optimizer_class' : optimizer_class,
                                                            'learning_rate' : learning_rate,
                                                            'batch_size' : batch_size,
                                                            'activation' : activation,
                                                            'dropout_rate' : dropout_rate,
                                                            'accuracy_rate' : final_accuracy_rate,

                                                }
                                    scores.append(score)
                                    print("******************************************")
                                    print(scores[-1])
                                    print("******************************************")

                                    with open("search_results/gridSearchDNN_results.txt","a") as f:
                                        f.write(str(score) + "\n")

        best_score = 0
        for score in scores:
            accuracy = score['accuracy_rate']
            if accuracy > best_score:
                self.best_params = score
        print("Best parameters:")
        print(self.best_params)