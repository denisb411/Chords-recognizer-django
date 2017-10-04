from .tensorflow_models import CNNClassifier, DNNClassifier

class RandomSearchCNN(object):
    def __init__(self, params, k_fold=5, num_random_combinations=100):
        """GridSearch on CNN with cross_validation with k_fold = 5"""
        self.best_params = None
        self.k_fold = k_fold
        self.params = {
                        'n_hidden_layers':params['n_hidden_layers'],
                        'n_neurons':params['n_neurons'],
                        'optimizer_class':params['optimizer_class'],
                        'learning_rate':params['learning_rate'],
                        'dropout_rate':params['dropout_rate'],
                        'batch_size':params['batch_size'],
                        'activation':params['activation'],
                        'conv1':params['conv1'],
                        'conv2':params['conv2'],
                        'architecture':params['architecture']
        }

        max_indexes = [len(v) for k, v in self.params.items()]
        num_random_combinations = min(num_random_combinations, np.prod(max_indexes))

        # generate unique combinations
        combinations = set()
        while len(combinations) < num_random_combinations:
            combinations.add(tuple(
                random.randint(0, max_index - 1)
                for max_index in max_indexes))
        # make sure their order is shuffled
        # (`set` seems to sort its content)
        combinations = list(combinations)
        random.shuffle(combinations)

        self.combinations = combinations

    def fit(self, X_train, y_train, X_test=None, y_test=None, X_valid=None, y_valid=None):

        scores = []
        print("testing", len(self.combinations), "combinations.")
        for combination in self.combinations:
            accuracy_rate = 0
            folds = self.k_fold
            if folds <= 1:
                if X_test == None or y_test == None:
                    raise ValueError("Pass the test set when using kfold = 1!")

                print("Trining CNN with parameters: " + 
                        "n_hidden_layers: %d, " % (self.params['n_hidden_layers'][combination[0]]) +
                        "n_neurons: %d, " % (self.params['n_neurons'][combination[1]]) +
                        "optimizer_class: %s, " % (self.params['optimizer_class'][combination[2]]) +
                        "learning_rate: %d, " % (self.params['learning_rate'][combination[3]]) +
                        "dropout_rate: %d, " % (self.params['dropout_rate'][combination[4]]) +
                        "batch_size: %d, " % (self.params['batch_size'][combination[5]]) +
                        "activation: %s, " % (self.params['activation'][combination[6]]) +
                        "conv1: %s, " % (self.params['conv1'][combination[7]]) +
                        "conv2: %s ." % (self.params['conv2'][combination[8]]) +
                        "architecture: %d, " % (self.params['architecture'][combination[9]])
                    )

                n_hidden_layers = self.params['n_hidden_layers'][combination[0]]
                n_neurons = self.params['n_neurons'][combination[1]]
                optimizer_class = self.params['optimizer_class'][combination[2]]
                learning_rate = self.params['learning_rate'][combination[3]]
                dropout_rate = self.params['dropout_rate'][combination[4]]
                batch_size = self.params['batch_size'][combination[5]]
                activation = self.params['activation'][combination[6]]
                conv1 = self.params['conv1'][combination[7]]
                conv2 = self.params['conv2'][combination[8]]
                architecture = self.params['architecture'][combination[9]]

                cnn = CNNClassifier(n_hidden_layers=n_hidden_layers, n_neurons=n_neurons, 
                    optimizer_class=optimizer_class,
                    learning_rate=learning_rate, batch_size=batch_size, 
                    dropout_rate=dropout_rate, activation=activation, 
                    conv1=conv1, conv2=conv2, architecture=architecture)

                cnn.fit(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid)
                accuracy_rate += cnn.accuracy_score(X_test, y_test)
                del cnn

            else:
                k_fold_samples = int(len(X) / folds)
                try:
                    X = np.append(X_train, X_test)
                    y = np.append(y_train, y_test)
                except:
                    pass
                print("Trining CNN with parameters: " + 
                            "n_hidden_layers: %d, " % (self.params['n_hidden_layers'][combination[0]]) +
                            "n_neurons: %d, " % (self.params['n_neurons'][combination[1]]) +
                            "optimizer_class: %s, " % (self.params['optimizer_class'][combination[2]]) +
                            "learning_rate: %d, " % (self.params['learning_rate'][combination[3]]) +
                            "dropout_rate: %d, " % (self.params['dropout_rate'][combination[4]]) +
                            "batch_size: %d, " % (self.params['batch_size'][combination[5]]) +
                            "activation: %s, " % (self.params['activation'][combination[6]]) +
                            "conv1: %s, " % (self.params['conv1'][combination[7]]) +
                            "conv2: %s ." % (self.params['conv2'][combination[8]]) +
                            "architecture: %d, " % (self.params['architecture'][combination[9]])
                        )
                for k in range(folds):
                    n_hidden_layers = self.params['n_hidden_layers'][combination[0]]
                    n_neurons = self.params['n_neurons'][combination[1]]
                    optimizer_class = self.params['optimizer_class'][combination[2]]
                    learning_rate = self.params['learning_rate'][combination[3]]
                    dropout_rate = self.params['dropout_rate'][combination[4]]
                    batch_size = self.params['batch_size'][combination[5]]
                    activation = self.params['activation'][combination[6]]
                    conv1 = self.params['conv1'][combination[7]]
                    conv2 = self.params['conv2'][combination[8]]
                    architecture = self.params['architecture'][combination[9]]

                    print("Training and testing fold %i" % k)
                    range1 = k_fold_samples * ((folds - 1) - k)
                    range2 = k_fold_samples * (folds - k)

                    print("ranges:")
                    print(range1, range2)

                    X_train_step = X[:range1]
                    X_train_step = np.vstack([X_train, X[range2:]])
                    y_train_step = y[:range1]
                    y_train_step = np.append(y_train, y[range2:])

                    X_test_step = X[range1:range2]
                    y_test_step = y[range1:range2]

                    print("shapes:")
                    print(X_train_step.shape, X_test_step.shape)
                    print(y_train_step.shape, y_test_step.shape)

                    cnn = CNNClassifier(n_hidden_layers=n_hidden_layers, n_neurons=n_neurons, 
                        optimizer_class=optimizer_class,
                        learning_rate=learning_rate, batch_size=batch_size, 
                        dropout_rate=dropout_rate, activation=activation, 
                        conv1=conv1, conv2=conv2, architecture=architecture)

                    cnn.fit(X_train=X_train_step, y_train=y_train_step, X_valid=X_valid, y_valid=y_valid)
                    accuracy_rate += cnn.accuracy_score(X_test_step, y_test_step)

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
            print(score)
            print("******************************************")

            with open("search_results/randomSearchCNN_results.txt","a") as f:
                f.write(str(score) + "\n")

        best_score = 0
        for score in scores:
            accuracy = score['accuracy_rate']
            if accuracy > best_score:
                best_score = accuracy
                self.best_params = score

        print("Best parameters:")
        print(self.best_params)

class RandomSearchDNN(object):
    def __init__(self, params, k_fold=5, num_random_combinations=100):
        """GridSearch on CNN with cross_validation with k_fold = 5"""
        self.best_params = None
        self.k_fold = k_fold
        self.params = {
                        'n_hidden_layers':params['n_hidden_layers'],
                        'n_neurons':params['n_neurons'],
                        'optimizer_class':params['optimizer_class'],
                        'learning_rate':params['learning_rate'],
                        'dropout_rate':params['dropout_rate'],
                        'batch_size':params['batch_size'],
                        'activation':params['activation'],
        }

        max_indexes = [len(v) for k, v in self.params.items()]
        num_random_combinations = min(num_random_combinations, np.prod(max_indexes))

        # generate unique combinations
        combinations = set()
        while len(combinations) < num_random_combinations:
            combinations.add(tuple(
                random.randint(0, max_index - 1)
                for max_index in max_indexes))
        # make sure their order is shuffled
        # (`set` seems to sort its content)
        combinations = list(combinations)
        random.shuffle(combinations)

        self.combinations = combinations


    def fit(self, X, y, X_valid=None, y_valid=None):

        scores = []
        for combination in self.combinations:
            accuracy_rate = 0
            folds = self.k_fold
            if folds <= 1:
                print("Trining DNN with parameters: " + 
                        "n_hidden_layers: %d, " % (self.params['n_hidden_layers'][combination[0]]) +
                        "n_neurons: %d, " % (self.params['n_neurons'][combination[1]]) +
                        "optimizer_class: %s, " % (self.params['optimizer_class'][combination[2]]) +
                        "learning_rate: %d, " % (self.params['learning_rate'][combination[3]]) +
                        "dropout_rate: %d, " % (self.params['dropout_rate'][combination[4]]) +
                        "batch_size: %d, " % (self.params['batch_size'][combination[5]]) +
                        "activation: %s, " % (self.params['activation'][combination[6]])
                    )

                n_hidden_layers = self.params['n_hidden_layers'][combination[0]]
                n_neurons = self.params['n_neurons'][combination[1]]
                optimizer_class = self.params['optimizer_class'][combination[2]]
                learning_rate = self.params['learning_rate'][combination[3]]
                dropout_rate = self.params['dropout_rate'][combination[4]]
                batch_size = self.params['batch_size'][combination[5]]
                activation = self.params['activation'][combination[6]]

                trainRange = int(len(X) * 0.85)
                testRange = int(len(X) * 0.15)

                X_train = X[:trainRange]
                y_train = y[:trainRange]

                X_test = X[:testRange]
                y_test = y[:testRange]

                print("shapes:")
                print(X_train.shape, X_test.shape)
                print(y_train.shape, y_test.shape)

                dnn = DNNClassifier(n_hidden_layers=n_hidden_layers, n_neurons=n_neurons, optimizer_class=optimizer_class,
                                    learning_rate=learning_rate, activation=activation, batch_size=batch_size, dropout_rate=dropout_rate)

                dnn.fit(X=X_train, y=y_train, X_valid=X_valid, y_valid=y_valid)
                accuracy_rate += dnn.accuracy_score(X_test, y_test)
                del dnn

            else:
                k_fold_samples = int(len(X) / folds)
                try:
                    X = np.append(X_train, X_test)
                    y = np.append(y_train, y_test)
                except:
                    pass
                print("Trining DNN with parameters: " + 
                        "n_hidden_layers: %d, " % (self.params['n_hidden_layers'][combination[0]]) +
                        "n_neurons: %d, " % (self.params['n_neurons'][combination[1]]) +
                        "optimizer_class: %s, " % (self.params['optimizer_class'][combination[2]]) +
                        "learning_rate: %d, " % (self.params['learning_rate'][combination[3]]) +
                        "dropout_rate: %d, " % (self.params['dropout_rate'][combination[4]]) +
                        "batch_size: %d, " % (self.params['batch_size'][combination[5]]) +
                        "activation: %s, " % (self.params['activation'][combination[6]])
                    )
                for k in range(folds):
                    n_hidden_layers = self.params['n_hidden_layers'][combination[0]]
                    n_neurons = self.params['n_neurons'][combination[1]]
                    optimizer_class = self.params['optimizer_class'][combination[2]]
                    learning_rate = self.params['learning_rate'][combination[3]]
                    dropout_rate = self.params['dropout_rate'][combination[4]]
                    batch_size = self.params['batch_size'][combination[5]]
                    activation = self.params['activation'][combination[6]]

                    print("Training and testing fold %i" % k)
                    range1 = k_fold_samples * ((folds - 1) - k)
                    range2 = k_fold_samples * (folds - k)

                    print("ranges:")
                    print(range1, range2)

                    X_train_step = X[:range1]
                    X_train_step = np.vstack([X_train, X[range2:]])
                    y_train_step = y[:range1]
                    y_train_step = np.append(y_train, y[range2:])

                    X_test_step = X[range1:range2]
                    y_test_step = y[range1:range2]

                    print("shapes:")
                    print(X_train_step.shape, X_test_step.shape)
                    print(y_train_step.shape, y_test_step.shape)

                    dnn = DNNClassifier(n_hidden_layers=n_hidden_layers, n_neurons=n_neurons, optimizer_class=optimizer_class,
                                        learning_rate=learning_rate, activation=activation, batch_size=batch_size, dropout_rate=dropout_rate)

                    dnn.fit(X=X_train_step, y=y_train_step, X_valid=X_valid, y_valid=y_valid)
                    accuracy_rate += dnn.accuracy_score(X_test_step, y_test_step)
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

            with open("search_results/randomSearchDNN_results.txt","a") as f:
                f.write(str(score) + "\n")

        best_score = 0
        for score in scores:
            accuracy = score['accuracy_rate']
            if accuracy > best_score:
                best_score = accuracy
                self.best_params = score
                
        print("Best parameters:")
        print(self.best_params)